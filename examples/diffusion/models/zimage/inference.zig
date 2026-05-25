const std = @import("std");
const zml = @import("zml");
const zigimg = @import("zigimg");

const common = @import("../common.zig");
const zimage_model = @import("model.zig");
const zimage_text_encoder = @import("text_encoder.zig");
const zimage_tokenizer = @import("tokenizer.zig");
const zimage_scheduler = @import("scheduler.zig");
const zimage_transformer = @import("transformer.zig");
const zimage_vae = @import("vae.zig");

const log = std.log.scoped(.zimage);

fn ensurePngOutputPath(allocator: std.mem.Allocator, output_path: []const u8) ![]const u8 {
    if (std.mem.endsWith(u8, output_path, ".png")) {
        return allocator.dupe(u8, output_path);
    }

    const ext = std.fs.path.extension(output_path);
    if (ext.len > 0) {
        const stem = output_path[0 .. output_path.len - ext.len];
        return std.fmt.allocPrint(allocator, "{s}.png", .{stem});
    }

    return std.fmt.allocPrint(allocator, "{s}.png", .{output_path});
}

pub const InferenceErrors = error{
    MissingPrompt,
    InvalidImageSize,
    PromptTooLong,
    InvalidDecodedImage,
};

pub const RunOptions = struct {
    prompt: []const u8,
    negative_prompt: []const u8 = "",
    height: u32 = 1024,
    width: u32 = 1024,
    num_inference_steps: u32 = 50,
    guidance_scale: f32 = 5.0,
    cfg_normalization: bool = false,
    seed: u64 = 0,
    output_path: []const u8,
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
    scheduler_step: zml.Exe,
    vae_decode: zml.Exe,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded_model: *const zimage_model.LoadedModel,
        params: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const compiled_result = try compile_whole_model(allocator, io, platform, loaded_model, params, progress);
        return .{
            .loaded_model = loaded_model,
            .params = params,

            .text_encoder = compiled_result.text_encoder,
            .transformer = compiled_result.transformer,
            .scheduler_step = compiled_result.scheduler_step,
            .vae_decode = compiled_result.vae_decode,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.text_encoder.deinit();
        self.transformer.deinit();
        self.scheduler_step.deinit();
        self.vae_decode.deinit();
    }
};

pub const CompiledModelResult = struct {
    text_encoder: zml.Exe,
    transformer: zml.Exe,
    scheduler_step: zml.Exe,
    vae_decode: zml.Exe,
};

const F32Buffer = struct {
    buffer: zml.Buffer,
    owned: bool,
};

const BufferStats = struct {
    min: f32,
    max: f32,
    mean: f64,

    fn fmt(self: BufferStats, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("min={d:.6} max={d:.6} mean={d:.6}", .{ self.min, self.max, self.mean });
    }
};

const PromptInputs = struct {
    ids: zml.Buffer,
    mask: zml.Buffer,
    used_tokens: u32,

    fn deinit(self: *PromptInputs) void {
        self.ids.deinit();
        self.mask.deinit();
    }
};

const EncodedPrompt = struct {
    embeds: zml.Buffer,
    seq_len: u32,

    fn deinit(self: *EncodedPrompt) void {
        self.embeds.deinit();
    }
};

fn bufferToF32(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    buffer: zml.Buffer,
) !F32Buffer {
    if (buffer.shape().dtype() == .f32) {
        return .{
            .buffer = buffer,
            .owned = false,
        };
    }

    var src = try buffer.toSliceAlloc(allocator, io);
    defer src.free(allocator);

    var dst = try zml.Slice.alloc(allocator, buffer.shape().withDtype(.f32));
    errdefer dst.free(allocator);

    switch (src.dtype()) {
        inline else => |dt| {
            const src_items = src.constItems(dt.toZigType());
            const dst_items = dst.items(f32);
            for (src_items, dst_items) |value, *out| {
                out.* = switch (comptime dt.class()) {
                    .float => switch (dt) {
                        else => zml.floats.floatCast(f32, value),
                    },
                    .integer => @floatFromInt(value),
                    .bool => if (value) 1.0 else 0.0,
                    else => unreachable,
                };
            }
        },
    }

    const converted = try zml.Buffer.fromSlice(io, platform, dst, .replicated);
    dst.free(allocator);
    return .{
        .buffer = converted,
        .owned = true,
    };
}

fn computeSliceStats(slice: zml.Slice) BufferStats {
    return switch (slice.dtype()) {
        inline else => |dt| blk: {
            const values = slice.constItems(dt.toZigType());
            if (values.len == 0) break :blk BufferStats{ .min = 0.0, .max = 0.0, .mean = 0.0 };

            var min_value: f32 = std.math.inf(f32);
            var max_value: f32 = -std.math.inf(f32);
            var sum: f64 = 0.0;
            for (values) |value| {
                const as_f32: f32 = switch (comptime dt.class()) {
                    .float => zml.floats.floatCast(f32, value),
                    .integer => @floatFromInt(value),
                    .bool => if (value) 1.0 else 0.0,
                    else => unreachable,
                };
                min_value = @min(min_value, as_f32);
                max_value = @max(max_value, as_f32);
                sum += as_f32;
            }

            break :blk BufferStats{
                .min = min_value,
                .max = max_value,
                .mean = sum / @as(f64, @floatFromInt(values.len)),
            };
        },
    };
}

fn logBufferStats(
    allocator: std.mem.Allocator,
    io: std.Io,
    label: []const u8,
    buffer: zml.Buffer,
) !void {
    var slice = try buffer.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    log.info("{s}: {}", .{ label, computeSliceStats(slice) });
}

fn logStepStats(
    allocator: std.mem.Allocator,
    io: std.Io,
    step_index: usize,
    comptime label_fmt: []const u8,
    buffer: zml.Buffer,
) !void {
    const label = try std.fmt.allocPrint(allocator, label_fmt, .{step_index});
    defer allocator.free(label);
    try logBufferStats(allocator, io, label, buffer);
}

fn compileTextEncoderExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    text_encoder: *const zimage_text_encoder.Qwen3ForCausalLM,
    prompt_seqlen: u32,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const input_ids: zml.Tensor = .init(.{ .b = 1, .s = prompt_seqlen }, .u32);
    const attention_mask: zml.Tensor = .init(.{ .b = 1, .s = prompt_seqlen }, .bool);
    return platform.compileModel(
        allocator,
        io,
        zimage_text_encoder.Qwen3ForCausalLM.encodePrompt,
        text_encoder.*,
        .{ input_ids, attention_mask },
        .{ .shardings = shardings },
    );
}

fn compileTransformerExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    transformer: *const zimage_transformer.Transformer,
    prompt_seqlen: u32,
    latent_frames: u32,
    latent_height: u32,
    latent_width: u32,
    hidden_size: u32,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const latent: zml.Tensor = .init(.{
        .c = transformer.in_channels,
        .f = latent_frames,
        .h = latent_height,
        .w = latent_width,
    }, .f32);
    const timestep: zml.Tensor = .init(.{ .b = 1 }, .f32);
    const prompt_embeds: zml.Tensor = .init(.{
        .s = prompt_seqlen,
        .d = hidden_size,
    }, .f32);

    return platform.compileFn(
        allocator,
        io,
        zimage_transformer.Transformer.denoiseStep,
        .{ transformer, latent, timestep, prompt_embeds },
        .{ .shardings = shardings },
    );
}

fn createPromptInputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    tokenizer: *const zimage_tokenizer.Tokenizer,
    prompt_seqlen: u32,
    prompt_text: []const u8,
) !PromptInputs {
    const rendered_prompt = try tokenizer.applyUserChatTemplateAlloc(allocator, prompt_text);
    defer allocator.free(rendered_prompt);

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const prompt_tokens = try encoder.encodeAlloc(allocator, rendered_prompt);
    defer allocator.free(prompt_tokens);

    if (prompt_tokens.len > prompt_seqlen) return InferenceErrors.PromptTooLong;

    const token_shape = zml.Shape.init(.{ .b = 1, .s = prompt_seqlen }, .u32);
    var token_slice = try zml.Slice.alloc(allocator, token_shape);
    defer token_slice.free(allocator);
    const token_items = token_slice.items(u32);
    @memset(token_items, tokenizer.padTokenId() orelse 0);
    @memcpy(token_items[0..prompt_tokens.len], prompt_tokens);

    const mask_shape = zml.Shape.init(.{ .b = 1, .s = prompt_seqlen }, .bool);
    var mask_slice = try zml.Slice.alloc(allocator, mask_shape);
    defer mask_slice.free(allocator);
    const mask_items = mask_slice.items(bool);
    @memset(mask_items, false);
    @memset(mask_items[0..prompt_tokens.len], true);

    var ids = try zml.Buffer.fromSlice(io, platform, token_slice, .replicated);
    errdefer ids.deinit();
    const mask = try zml.Buffer.fromSlice(io, platform, mask_slice, .replicated);

    return .{
        .ids = ids,
        .mask = mask,
        .used_tokens = @intCast(prompt_tokens.len),
    };
}

fn extractPromptEmbeds(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    encoded: zml.Buffer,
    used_tokens: u32,
) !EncodedPrompt {
    var src = try encoded.toSliceAlloc(allocator, io);
    defer src.free(allocator);

    const hidden_size: u32 = @intCast(src.shape.dim(.d));
    var dst = try zml.Slice.alloc(allocator, zml.Shape.init(.{ .s = used_tokens, .d = hidden_size }, .f32));
    errdefer dst.free(allocator);

    const src_items = src.constItems(f32);
    const dst_items = dst.items(f32);
    const hidden_size_usize: usize = @intCast(hidden_size);
    const used_tokens_usize: usize = @intCast(used_tokens);
    for (0..used_tokens_usize) |s| {
        const src_offset = s * hidden_size_usize;
        const dst_offset = s * hidden_size_usize;
        @memcpy(dst_items[dst_offset .. dst_offset + hidden_size_usize], src_items[src_offset .. src_offset + hidden_size_usize]);
    }

    const embeds = try zml.Buffer.fromSlice(io, platform, dst, .replicated);
    dst.free(allocator);
    return .{
        .embeds = embeds,
        .seq_len = used_tokens,
    };
}

fn compileSchedulerStepExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    latent_channels: u32,
    latent_frames: u32,
    latent_height: u32,
    latent_width: u32,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const model_output: zml.Tensor = .init(.{
        .c = latent_channels,
        .f = latent_frames,
        .h = latent_height,
        .w = latent_width,
    }, .f32);
    const sample: zml.Tensor = .init(.{
        .c = latent_channels,
        .f = latent_frames,
        .h = latent_height,
        .w = latent_width,
    }, .f32);
    const current_sigma: zml.Tensor = .init(.{}, .f32);
    const next_sigma: zml.Tensor = .init(.{}, .f32);
    const kernel: zimage_model.SchedulerStepKernel = .{};

    return platform.compile(
        allocator,
        io,
        kernel,
        .step,
        .{ model_output, sample, current_sigma, next_sigma },
        .{ .shardings = shardings },
    );
}

fn compileVaeDecodeExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    vae: *const zimage_vae.AutoEncoder,
    latent_frames: u32,
    latent_height: u32,
    latent_width: u32,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const latents: zml.Tensor = .init(.{
        .c = vae.config.latent_channels,
        .f = latent_frames,
        .h = latent_height,
        .w = latent_width,
    }, .f32);

    return platform.compileModel(
        allocator,
        io,
        zimage_vae.AutoEncoder.decodeLatents,
        vae,
        .{latents},
        .{ .shardings = shardings },
    );
}

pub fn compile_whole_model(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    loaded_model: *const zimage_model.LoadedModel,
    params: CompilationParameters,
    progress: *std.Progress.Node,
) !CompiledModelResult {
    log.info("compile_whole_model: entry", .{});
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled model [{f}]", .{now.untilNow(io, .awake)});

    log.info("compile_whole_model: before shardings", .{});
    const all_shardings = params.shardings.all();
    log.info("compile_whole_model: after shardings", .{});
    const text_encoder = &loaded_model.inner.text_encoder.inner;
    log.info("compile_whole_model: text encoder ref ready", .{});
    const transformer = &loaded_model.inner.transformer;
    log.info("compile_whole_model: transformer ref ready", .{});
    const vae = &loaded_model.inner.vae;
    log.info("compile_whole_model: vae ref ready", .{});
    progress.increaseEstimatedTotalItems(4);

    const text_encoder_exe = blk: {
        var node = progress.start("Compiling text encoder...", 1);
        defer node.end();
        log.info("Starting text encoder compile...", .{});
        const exe = try compileTextEncoderExe(allocator, io, platform, text_encoder, params.prompt_seqlen, &all_shardings);
        log.info("Finished text encoder compile.", .{});
        break :blk exe;
    };
    errdefer text_encoder_exe.deinit();

    const transformer_exe = blk: {
        var node = progress.start("Compiling transformer...", 1);
        defer node.end();
        log.info("Starting transformer compile...", .{});
        const exe = try compileTransformerExe(
            allocator,
            io,
            platform,
            transformer,
            params.prompt_seqlen,
            params.latent_frames,
            params.latent_height,
            params.latent_width,
            loaded_model.inner.text_encoder.config.hidden_size,
            &all_shardings,
        );
        log.info("Finished transformer compile.", .{});
        break :blk exe;
    };
    errdefer transformer_exe.deinit();

    const scheduler_step_exe = blk: {
        var node = progress.start("Compiling scheduler step...", 1);
        defer node.end();
        log.info("Starting scheduler step compile...", .{});
        const exe = try compileSchedulerStepExe(
            allocator,
            io,
            platform,
            vae.config.latent_channels,
            params.latent_frames,
            params.latent_height,
            params.latent_width,
            &all_shardings,
        );
        log.info("Finished scheduler step compile.", .{});
        break :blk exe;
    };
    errdefer scheduler_step_exe.deinit();

    const vae_decode_exe = blk: {
        var node = progress.start("Compiling VAE decode...", 1);
        defer node.end();
        log.info("Starting VAE decode compile...", .{});
        const exe = try compileVaeDecodeExe(
            allocator,
            io,
            platform,
            vae,
            params.latent_frames,
            params.latent_height,
            params.latent_width,
            &all_shardings,
        );
        log.info("Finished VAE decode compile.", .{});
        break :blk exe;
    };

    return .{
        .text_encoder = text_encoder_exe,
        .transformer = transformer_exe,
        .scheduler_step = scheduler_step_exe,
        .vae_decode = vae_decode_exe,
    };
}

pub const InferencePipeline = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled_model: CompiledModel,
    model_buffers: *zimage_model.Buffers,
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

        const model_buffers = try allocator.create(zimage_model.Buffers);
        errdefer allocator.destroy(model_buffers);
        try loaded_model.loadBuffers(
            allocator,
            io,
            platform,
            store,
            progress,
            shardings,
            model_buffers,
        );
        errdefer loaded_model.unloadBuffers(model_buffers, allocator);

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
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .compiled_model = compiled_model,
            .model_buffers = model_buffers,
            .text_tokenizer = text_tokenizer,
            .scheduler_state = scheduler_state,
        };
    }

    pub fn deinit(self: *InferencePipeline, allocator: std.mem.Allocator) void {
        self.compiled_model.loaded_model.unloadBuffers(self.model_buffers, allocator);
        allocator.destroy(self.model_buffers);
        self.compiled_model.deinit();
        self.text_tokenizer.deinit();
        self.scheduler_state.deinit(allocator);
    }

    pub fn run(self: *InferencePipeline, opts: RunOptions) !void {
        if (opts.prompt.len == 0) return InferenceErrors.MissingPrompt;
        const do_cfg = opts.guidance_scale > 0.0;

        if (opts.height % 16 != 0 or opts.width % 16 != 0) return InferenceErrors.InvalidImageSize;

        const latent_height = opts.height / 8;
        const latent_width = opts.width / 8;
        if (latent_height != self.compiled_model.params.latent_height or latent_width != self.compiled_model.params.latent_width) {
            return InferenceErrors.InvalidImageSize;
        }

        var prompt_inputs = try createPromptInputs(
            self.allocator,
            self.io,
            self.platform,
            &self.text_tokenizer,
            self.compiled_model.params.prompt_seqlen,
            opts.prompt,
        );
        defer prompt_inputs.deinit();

        var text_encoder_args = try self.compiled_model.text_encoder.args(self.allocator);
        defer text_encoder_args.deinit(self.allocator);
        var text_encoder_results = try self.compiled_model.text_encoder.results(self.allocator);
        defer text_encoder_results.deinit(self.allocator);

        text_encoder_args.set(.{ &self.model_buffers.text_encoder.inner, prompt_inputs.ids, prompt_inputs.mask });
        self.compiled_model.text_encoder.call(text_encoder_args, &text_encoder_results);

        var prompt_embeds_full = text_encoder_results.get(zml.Buffer);
        defer prompt_embeds_full.deinit();

        var prompt_embeds = try extractPromptEmbeds(
            self.allocator,
            self.io,
            self.platform,
            prompt_embeds_full,
            prompt_inputs.used_tokens,
        );
        defer prompt_embeds.deinit();

        var negative_prompt_embeds: ?EncodedPrompt = null;
        defer if (negative_prompt_embeds) |*encoded| encoded.deinit();

        if (do_cfg) {
            var negative_prompt_inputs = try createPromptInputs(
                self.allocator,
                self.io,
                self.platform,
                &self.text_tokenizer,
                self.compiled_model.params.prompt_seqlen,
                if (opts.negative_prompt.len == 0) "" else opts.negative_prompt,
            );
            defer negative_prompt_inputs.deinit();

            var negative_text_encoder_args = try self.compiled_model.text_encoder.args(self.allocator);
            defer negative_text_encoder_args.deinit(self.allocator);
            var negative_text_encoder_results = try self.compiled_model.text_encoder.results(self.allocator);
            defer negative_text_encoder_results.deinit(self.allocator);

            negative_text_encoder_args.set(.{
                &self.model_buffers.text_encoder.inner,
                negative_prompt_inputs.ids,
                negative_prompt_inputs.mask,
            });
            self.compiled_model.text_encoder.call(negative_text_encoder_args, &negative_text_encoder_results);

            var negative_prompt_embeds_full = negative_text_encoder_results.get(zml.Buffer);
            defer negative_prompt_embeds_full.deinit();

            negative_prompt_embeds = try extractPromptEmbeds(
                self.allocator,
                self.io,
                self.platform,
                negative_prompt_embeds_full,
                negative_prompt_inputs.used_tokens,
            );
        }

        const all_shardings = self.compiled_model.params.shardings.all();
        var transformer_exe = try compileTransformerExe(
            self.allocator,
            self.io,
            self.platform,
            &self.compiled_model.loaded_model.inner.transformer,
            prompt_embeds.seq_len,
            self.compiled_model.params.latent_frames,
            self.compiled_model.params.latent_height,
            self.compiled_model.params.latent_width,
            self.compiled_model.loaded_model.inner.text_encoder.config.hidden_size,
            &all_shardings,
        );
        defer transformer_exe.deinit();

        var negative_transformer_exe: ?zml.Exe = null;
        defer if (negative_transformer_exe) |*exe| exe.deinit();
        if (negative_prompt_embeds) |negative_encoded| {
            if (negative_encoded.seq_len != prompt_embeds.seq_len) {
                negative_transformer_exe = try compileTransformerExe(
                    self.allocator,
                    self.io,
                    self.platform,
                    &self.compiled_model.loaded_model.inner.transformer,
                    negative_encoded.seq_len,
                    self.compiled_model.params.latent_frames,
                    self.compiled_model.params.latent_height,
                    self.compiled_model.params.latent_width,
                    self.compiled_model.loaded_model.inner.text_encoder.config.hidden_size,
                    &all_shardings,
                );
            }
        }

        const latent_shape = zml.Shape.init(.{
            .c = self.compiled_model.loaded_model.inner.transformer.in_channels,
            .f = self.compiled_model.params.latent_frames,
            .h = self.compiled_model.params.latent_height,
            .w = self.compiled_model.params.latent_width,
        }, .f32);
        var latent_slice = try zml.Slice.alloc(self.allocator, latent_shape);
        defer latent_slice.free(self.allocator);
        var prng = std.Random.DefaultPrng.init(opts.seed);
        const random = prng.random();
        for (latent_slice.items(f32)) |*value| {
            value.* = random.floatNorm(f32);
        }

        var latents = try zml.Buffer.fromSlice(self.io, self.platform, latent_slice, .replicated);
        defer latents.deinit();
        try logBufferStats(self.allocator, self.io, "initial latents", latents);

        const image_seq_len = (self.compiled_model.params.latent_height / 2) * (self.compiled_model.params.latent_width / 2);
        const mu: ?f32 = if (self.scheduler_state.use_dynamic_shifting) blk: {
            const base_seq_len = @as(f32, @floatFromInt(self.scheduler_state.base_image_seq_len));
            const max_seq_len = @as(f32, @floatFromInt(self.scheduler_state.max_image_seq_len));
            const base_shift = self.scheduler_state.base_shift orelse 0.5;
            const max_shift = self.scheduler_state.max_shift orelse 1.15;
            const slope = (max_shift - base_shift) / (max_seq_len - base_seq_len);
            const intercept = base_shift - slope * base_seq_len;
            break :blk @as(f32, @floatFromInt(image_seq_len)) * slope + intercept;
        } else null;
        self.scheduler_state.sigma_min = 0.0;
        try self.scheduler_state.setTimesteps(self.allocator, opts.num_inference_steps, mu);

        for (self.scheduler_state.timesteps) |raw_timestep| {
            const step_index = self.scheduler_state.step_index orelse 0;
            const normalized_timestep = (1000.0 - raw_timestep) / 1000.0;
            const timestep_values = [_]f32{normalized_timestep};
            var timestep_buffer = try zml.Buffer.fromBytes(
                self.io,
                self.platform,
                zml.Shape.init(.{ .b = 1 }, .f32),
                .replicated,
                std.mem.sliceAsBytes(&timestep_values),
            );
            defer timestep_buffer.deinit();

            var transformer_args = try transformer_exe.args(self.allocator);
            defer transformer_args.deinit(self.allocator);
            var transformer_results = try transformer_exe.results(self.allocator);
            defer transformer_results.deinit(self.allocator);

            transformer_args.set(.{ &self.model_buffers.transformer, latents, timestep_buffer, prompt_embeds.embeds });
            transformer_exe.call(transformer_args, &transformer_results);

            var noise_pred = transformer_results.get(zml.Buffer);
            try logStepStats(self.allocator, self.io, step_index, "step {d} transformer output", noise_pred);
            var scheduler_input = noise_pred;

            if (do_cfg) {
                const negative_transformer_ref = if (negative_transformer_exe) |*exe| exe else &transformer_exe;
                var negative_transformer_args = try negative_transformer_ref.args(self.allocator);
                defer negative_transformer_args.deinit(self.allocator);
                var negative_transformer_results = try negative_transformer_ref.results(self.allocator);
                defer negative_transformer_results.deinit(self.allocator);

                negative_transformer_args.set(.{
                    &self.model_buffers.transformer,
                    latents,
                    timestep_buffer,
                    negative_prompt_embeds.?.embeds,
                });
                negative_transformer_ref.call(negative_transformer_args, &negative_transformer_results);

                var negative_noise_pred = negative_transformer_results.get(zml.Buffer);
                defer negative_noise_pred.deinit();
                try logStepStats(self.allocator, self.io, step_index, "step {d} negative transformer output", negative_noise_pred);

                var positive_f32 = try bufferToF32(self.allocator, self.io, self.platform, noise_pred);
                defer if (positive_f32.owned) positive_f32.buffer.deinit();
                var negative_f32 = try bufferToF32(self.allocator, self.io, self.platform, negative_noise_pred);
                defer if (negative_f32.owned) negative_f32.buffer.deinit();

                var positive_slice = try positive_f32.buffer.toSliceAlloc(self.allocator, self.io);
                defer positive_slice.free(self.allocator);
                var negative_slice = try negative_f32.buffer.toSliceAlloc(self.allocator, self.io);
                defer negative_slice.free(self.allocator);

                var positive_norm: f64 = 0.0;
                var pred_norm: f64 = 0.0;
                const positive_items = positive_slice.items(f32);
                const negative_items = negative_slice.constItems(f32);
                for (positive_items, negative_items) |*positive, negative| {
                    const pos = positive.*;
                    const pred = -(pos + opts.guidance_scale * (pos - negative));
                    positive.* = pred;
                    positive_norm += @as(f64, pos) * @as(f64, pos);
                    pred_norm += @as(f64, pred) * @as(f64, pred);
                }

                if (opts.cfg_normalization and pred_norm > 0.0) {
                    const max_new_norm = std.math.sqrt(positive_norm);
                    const actual_new_norm = std.math.sqrt(pred_norm);
                    if (actual_new_norm > max_new_norm) {
                        const scale = @as(f32, @floatCast(max_new_norm / actual_new_norm));
                        for (positive_items) |*value| {
                            value.* *= scale;
                        }
                    }
                }

                scheduler_input = try zml.Buffer.fromSlice(self.io, self.platform, positive_slice, .replicated);
                noise_pred.deinit();
            } else {
                var noise_slice = try noise_pred.toSliceAlloc(self.allocator, self.io);
                defer noise_slice.free(self.allocator);
                for (noise_slice.items(f32)) |*value| {
                    value.* = -value.*;
                }

                scheduler_input = try zml.Buffer.fromSlice(self.io, self.platform, noise_slice, .replicated);
                noise_pred.deinit();
            }

            try logStepStats(self.allocator, self.io, step_index, "step {d} scheduler input", scheduler_input);
            defer scheduler_input.deinit();
            const next_latents = try self.runSchedulerStep(scheduler_input, latents, raw_timestep);
            try logStepStats(self.allocator, self.io, step_index, "step {d} scheduler output latents", next_latents);

            latents.deinit();
            latents = next_latents;
        }

        var scaled_latent_slice = try latents.toSliceAlloc(self.allocator, self.io);
        defer scaled_latent_slice.free(self.allocator);
        log.info("final latents before vae scaling: {}", .{computeSliceStats(scaled_latent_slice)});
        const scaling_factor = self.compiled_model.loaded_model.inner.vae.config.scaling_factor;
        const shift_factor = self.compiled_model.loaded_model.inner.vae.config.shift_factor;
        for (scaled_latent_slice.items(f32)) |*value| {
            value.* = (value.* / scaling_factor) + shift_factor;
        }
        log.info("latents after vae scaling: {}", .{computeSliceStats(scaled_latent_slice)});

        var scaled_latents = try zml.Buffer.fromSlice(self.io, self.platform, scaled_latent_slice, .replicated);
        defer scaled_latents.deinit();

        var vae_args = try self.compiled_model.vae_decode.args(self.allocator);
        defer vae_args.deinit(self.allocator);
        var vae_results = try self.compiled_model.vae_decode.results(self.allocator);
        defer vae_results.deinit(self.allocator);

        vae_args.set(.{ &self.model_buffers.vae, scaled_latents });
        self.compiled_model.vae_decode.call(vae_args, &vae_results);

        var image_buffer = vae_results.get(zml.Buffer);
        defer image_buffer.deinit();
        try logBufferStats(self.allocator, self.io, "vae decoded image", image_buffer);

        var image_f32 = try bufferToF32(self.allocator, self.io, self.platform, image_buffer);
        defer if (image_f32.owned) image_f32.buffer.deinit();

        var image_slice = try image_f32.buffer.toSliceAlloc(self.allocator, self.io);
        defer image_slice.free(self.allocator);

        if (image_slice.shape.rank() != 4 or image_slice.shape.dim(.b) != 1 or image_slice.shape.dim(.c) < 3) {
            return InferenceErrors.InvalidDecodedImage;
        }

        const image_height: usize = @intCast(image_slice.shape.dim(.h));
        const image_width: usize = @intCast(image_slice.shape.dim(.w));
        const image_items = image_slice.constItems(f32);

        var pixels = try self.allocator.alloc(zigimg.color.Rgb24, image_width * image_height);
        for (0..image_height) |y| {
            for (0..image_width) |x| {
                var rgb: [3]u8 = undefined;
                for (0..3) |c| {
                    const idx = ((c * image_height + y) * image_width) + x;
                    const value = std.math.clamp(image_items[idx], -1.0, 1.0) * 0.5 + 0.5;
                    rgb[c] = @intFromFloat(std.math.clamp(value * 255.0, 0.0, 255.0));
                }
                pixels[y * image_width + x] = zigimg.color.Rgb24.from.rgb(rgb[0], rgb[1], rgb[2]);
            }
        }

        var image: zigimg.Image = .{
            .width = image_width,
            .height = image_height,
            .pixels = .{ .rgb24 = pixels },
            .animation = .{},
        };
        defer image.deinit(self.allocator);

        const output_path = try ensurePngOutputPath(self.allocator, opts.output_path);
        defer self.allocator.free(output_path);

        var write_buffer: [zigimg.io.DEFAULT_BUFFER_SIZE]u8 = undefined;
        try image.writeToFilePath(self.allocator, self.io, output_path, write_buffer[0..], .{ .png = .{} });
    }

    pub fn runSchedulerStep(
        self: *InferencePipeline,
        noise_pred: zml.Buffer,
        latents: zml.Buffer,
        timestep: f32,
    ) !zml.Buffer {
        const sigmas = self.scheduler_state.nextStepSigmas(timestep);

        var scheduler_noise_pred = try bufferToF32(self.allocator, self.io, self.platform, noise_pred);
        defer if (scheduler_noise_pred.owned) scheduler_noise_pred.buffer.deinit();

        var scheduler_latents = try bufferToF32(self.allocator, self.io, self.platform, latents);
        defer if (scheduler_latents.owned) scheduler_latents.buffer.deinit();

        var current_sigma_buf = try zml.Buffer.scalar(self.io, self.platform, sigmas.current, .f32);
        defer current_sigma_buf.deinit();

        var next_sigma_buf = try zml.Buffer.scalar(self.io, self.platform, sigmas.next, .f32);
        defer next_sigma_buf.deinit();

        var scheduler_args = try self.compiled_model.scheduler_step.args(self.allocator);
        defer scheduler_args.deinit(self.allocator);

        var scheduler_results = try self.compiled_model.scheduler_step.results(self.allocator);
        defer scheduler_results.deinit(self.allocator);

        scheduler_args.set(.{
            scheduler_noise_pred.buffer,
            scheduler_latents.buffer,
            current_sigma_buf,
            next_sigma_buf,
        });
        self.compiled_model.scheduler_step.call(scheduler_args, &scheduler_results);

        return scheduler_results.get(zml.Buffer);
    }
};
