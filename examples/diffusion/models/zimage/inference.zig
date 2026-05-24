const std = @import("std");
const zml = @import("zml");

const common = @import("../common.zig");
const zimage_model = @import("model.zig");
const zimage_tokenizer = @import("tokenizer.zig");
const zimage_scheduler = @import("scheduler.zig");

const log = std.log.scoped(.zimage);

pub const InferenceErrors = enum {
    UnknownError,
};

pub const CompilationParameters = struct {
    prompt_seqlen: u32,
    latent_frames: u32,
    latent_height: u32,
    latent_width: u32,
    shardings: common.Shardings,

    pub fn init(
        prompt_seqlen: u32,
        latent_frames: u32,
        latent_height: u32,
        latent_width: u32,
        shardings: common.Shardings,
    ) CompilationParameters {
        return .{
            .prompt_seqlen = prompt_seqlen,
            .latent_frames = latent_frames,
            .latent_height = latent_height,
            .latent_width = latent_width,
            .shardings = shardings,
        };
    }
};

pub const CompiledModel = struct {
    loaded_model: *const zimage_model.LoadedModel,
    params: CompilationParameters,

    // text_encoder->transformer->vae
    text_encoder: zml.Exe,
    transformer: zml.Exe,
    vae_decode: zml.Exe,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded_model: *const zimage_model.LoadedModel,
        zimage: zimage_model.Model,
        params: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const compiled_result = try compile_whole_model(allocator, io, platform, zimage, params, progress);
        return .{
            .loaded_model = loaded_model,
            .params = params,

            .text_encoder = compiled_result.text_encoder,
            .transformer = compiled_result.transformer,
            .vae_decode = compiled_result.vae_decode,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.text_encoder.deinit();
        self.transformer.deinit();
        self.vae_decode.deinit();
    }
};

pub const CompiledModelResult = struct {
    text_encoder: zml.Exe,
    transformer: zml.Exe,
    vae_decode: zml.Exe,
};

pub fn compile_whole_model(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    zimage: zimage_model.Model,
    params: CompilationParameters,
    progress: *std.Progress.Node,
) !CompiledModelResult {
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled model [{f}]", .{now.untilNow(io, .awake)});

    const all_shardings = params.shardings.all();

    var text_encoder_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            zimage_: zimage_model.Model,
            params_: CompilationParameters,
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling text encoder...", 1);
            defer node_.end();

            const input_ids: zml.Tensor = .init(
                .{ .b = 1, .s = params_.prompt_seqlen },
                .u32,
            );

            return platform_.compile(
                allocator_,
                io_,
                zimage_,
                .encodePrompt,
                .{input_ids},
                .{ .shardings = shardings_ },
            );
        }
    }.call, .{ allocator, io, platform, zimage, params, &all_shardings, progress });
    errdefer if (text_encoder_future.cancel(io)) |v| v.deinit() else |_| {};

    var transformer_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            zimage_: zimage_model.Model,
            params_: CompilationParameters,
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling transformer...", 1);
            defer node_.end();

            const latent: zml.Tensor = .init(.{
                .c = zimage_.transformer.in_channels,
                .f = params_.latent_frames,
                .h = params_.latent_height,
                .w = params_.latent_width,
            }, .f32);

            const timestep: zml.Tensor = .init(.{ .b = 1 }, .f32);

            const prompt_embeds: zml.Tensor = .init(.{
                .s = params_.prompt_seqlen,
                .d = zimage_.text_encoder.config.hidden_size,
            }, .f32);

            return platform_.compile(
                allocator_,
                io_,
                zimage_,
                .denoiseStep,
                .{
                    latent,
                    timestep,
                    prompt_embeds,
                },
                .{ .shardings = shardings_ },
            );
        }
    }.call, .{ allocator, io, platform, zimage, params, &all_shardings, progress });
    errdefer if (transformer_future.cancel(io)) |v| v.deinit() else |_| {};

    var vae_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            zimage_: zimage_model.Model,
            params_: CompilationParameters,
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling VAE decode...", 1);
            defer node_.end();

            const latents: zml.Tensor = .init(.{
                .c = zimage_.vae.config.latent_channels,
                .f = params_.latent_frames,
                .h = params_.latent_height,
                .w = params_.latent_width,
            }, .f32);

            return platform_.compile(
                allocator_,
                io_,
                zimage_,
                .decodeLatents,
                .{latents},
                .{ .shardings = shardings_ },
            );
        }
    }.call, .{ allocator, io, platform, zimage, params, &all_shardings, progress });
    errdefer if (vae_future.cancel(io)) |v| v.deinit() else |_| {};

    const text_encoder_exe = try text_encoder_future.await(io);
    const transformer_exe = try transformer_future.await(io);
    const vae_decode_exe = try vae_future.await(io);

    return .{
        .text_encoder = text_encoder_exe,
        .transformer = transformer_exe,
        .vae_decode = vae_decode_exe,
    };
}

pub const InferencePipeline = struct {
    compiled_model: CompiledModel,
    model_buffers: zimage_model.Buffers,
    text_tokenizer: zimage_tokenizer.Tokenizer,
    scheduler_state: zimage_scheduler.Scheduler,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        platform: *const zml.Platform,
        loaded_model: *const zimage_model.LoadedModel,
        store: *zml.io.TensorStore,
        shardings: common.Shardings,
        prompt_seqlen: u32,
        progress: *std.Progress.Node,
    ) !InferencePipeline {
        const backend = zml.attention.attention.Backend.auto(platform);

        var compiled_model = try loaded_model.compile(
            allocator,
            io,
            platform,
            backend,
            shardings,
            prompt_seqlen,
            progress,
        );
        errdefer compiled_model.deinit();

        var model_buffers = try loaded_model.loadBuffers(
            allocator,
            io,
            platform,
            store,
            progress,
            shardings,
        );
        errdefer loaded_model.unloadBuffers(&model_buffers, allocator);

        var text_tokenizer = try zimage_tokenizer.Tokenizer.fromDir(
            allocator,
            io,
            repo,
            .{},
        );
        errdefer text_tokenizer.deinit();

        var scheduler_state = try zimage_scheduler.Scheduler.init(
            allocator,
            loaded_model.scheduler_config.value,
        );
        errdefer scheduler_state.deinit(allocator);

        return .{
            .compiled_model = compiled_model,
            .model_buffers = model_buffers,
            .text_tokenizer = text_tokenizer,
            .scheduler_state = scheduler_state,
        };
    }

    pub fn deinit(self: *InferencePipeline, allocator: std.mem.Allocator) void {
        self.compiled_model.loaded_model.unloadBuffers(&self.model_buffers, allocator);
        self.compiled_model.deinit();
        self.text_tokenizer.deinit();
        self.scheduler_state.deinit(allocator);
    }
};
