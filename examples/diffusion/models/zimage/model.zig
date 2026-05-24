const std = @import("std");
const zml = @import("zml");

const common = @import("../common.zig");
const inference = @import("inference.zig");
const scheduler = @import("scheduler.zig");
const text_encoder = @import("text_encoder.zig");
const transformer = @import("transformer.zig");
const vae = @import("vae.zig");

const log = std.log.scoped(.@"diffusion/zimage");

pub const Config = struct {
    text_encoder: text_encoder.TextEncoder.Config,
    transformer: transformer.Transformer.Config,
    vae: vae.AutoEncoder.Config,
    scheduler: scheduler.SchedulerConfig,
};

pub const Buffers = zml.Bufferized(Model);

pub const Model = struct {
    text_encoder: text_encoder.TextEncoder,
    transformer: transformer.Transformer,
    vae: vae.AutoEncoder,

    pub fn init(
        self: *Model,
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: *const Config,
    ) !void {
        log.info("Initializing text encoder model graph...", .{});
        const text_encoder_model = try text_encoder.TextEncoder.init(allocator, store.withPrefix("text_encoder"), config.text_encoder);
        errdefer text_encoder_model.deinit(allocator);

        log.info("Initializing transformer model graph...", .{});
        const transformer_model = try transformer.Transformer.init(allocator, store.withPrefix("transformer"), config.transformer);
        errdefer transformer_model.deinit(allocator);

        log.info("Initializing VAE model graph...", .{});
        const vae_model = try vae.AutoEncoder.init(store.withPrefix("vae"), config.vae);
        errdefer vae_model.deinit(allocator);

        self.* = .{
            .text_encoder = text_encoder_model,
            .transformer = transformer_model,
            .vae = vae_model,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.text_encoder.deinit(allocator);
        self.transformer.deinit(allocator);
        self.vae.deinit(allocator);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        text_encoder.Qwen3Model.unloadBuffers(&self.text_encoder.inner.model, allocator);
        self.text_encoder.inner.lm_head.weight.deinit();
        transformer.Transformer.unloadBuffers(&self.transformer);
        vae.AutoEncoder.unloadBuffers(&self.vae);
    }

    pub fn encodePrompt(self: Model, input_ids: zml.Tensor) zml.Tensor {
        return self.text_encoder.inner.forwardHidden(input_ids).squeeze(.b);
    }

    pub fn denoiseStep(self: Model, latent: zml.Tensor, timestep: zml.Tensor, prompt_embeds: zml.Tensor) zml.Tensor {
        return self.transformer.forward(
            latent,
            timestep,
            prompt_embeds,
            self.transformer.all_patch_size[0],
            self.transformer.all_f_patch_size[0],
        );
    }

    pub fn decodeLatents(self: Model, latents: zml.Tensor) zml.Tensor {
        return self.vae.decodeInternal(latents);
    }
};

pub const LoadedModel = struct {
    inner: Model,
    scheduler_state: scheduler.Scheduler,
    text_encoder_config: std.json.Parsed(text_encoder.TextEncoder.Config),
    transformer_config: std.json.Parsed(transformer.Transformer.Config),
    vae_config: std.json.Parsed(vae.AutoEncoder.Config),
    scheduler_config: std.json.Parsed(scheduler.SchedulerConfig),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
    ) !LoadedModel {
        log.info("Parsing text encoder config...", .{});
        const text_encoder_config = try common.parse_config_at_path(
            text_encoder.TextEncoder.Config,
            allocator,
            io,
            repo,
            "text_encoder/config.json",
        );
        errdefer text_encoder_config.deinit();
        log.info("Parsing transformer config...", .{});
        const transformer_config = try common.parse_config_at_path(
            transformer.Transformer.Config,
            allocator,
            io,
            repo,
            "transformer/config.json",
        );
        errdefer transformer_config.deinit();
        log.info("Parsing VAE config...", .{});
        const vae_config = try common.parse_config_at_path(
            vae.AutoEncoder.Config,
            allocator,
            io,
            repo,
            "vae/config.json",
        );
        errdefer vae_config.deinit();
        log.info("Parsing scheduler config...", .{});
        const scheduler_config = try parseSchedulerConfig(allocator, io, repo);
        errdefer scheduler_config.deinit();

        log.info("Building merged Z-Image config...", .{});
        const merged_config: Config = .{
            .text_encoder = text_encoder_config.value,
            .transformer = transformer_config.value,
            .vae = vae_config.value,
            .scheduler = scheduler_config.value,
        };

        log.info("Initializing merged Z-Image model...", .{});
        var inner: Model = undefined;
        try inner.init(allocator, store, &merged_config);

        return .{
            .inner = inner,
            .scheduler_state = try .init(allocator, scheduler_config.value),
            .text_encoder_config = text_encoder_config,
            .transformer_config = transformer_config,
            .vae_config = vae_config,
            .scheduler_config = scheduler_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.scheduler_state.deinit(allocator);
        self.text_encoder_config.deinit();
        self.transformer_config.deinit();
        self.vae_config.deinit();
        self.scheduler_config.deinit();
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !Buffers {
        progress.increaseEstimatedTotalItems(store.view().count());
        return zml.io.load(Model, &self.inner, allocator, io, platform, store, .{
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .shardings = &shardings.all(),
            .parallelism = 16,
        });
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self;
        Model.unloadBuffers(buffers, allocator);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.attention.Backend,
        shardings: common.Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        _ = backend;
        const params = inference.CompilationParameters.init(
            @intCast(seqlen),
            20,
            128,
            128,
            shardings,
        );
        return inference.CompiledModel.init(allocator, io, platform, self, self.inner, params, progress);
    }
};

fn parseSchedulerConfig(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo: std.Io.Dir,
) !std.json.Parsed(scheduler.SchedulerConfig) {
    const file = try repo.openFile(io, "scheduler/scheduler_config.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(scheduler.SchedulerConfig, allocator, &reader, .{ .ignore_unknown_fields = true });
}
