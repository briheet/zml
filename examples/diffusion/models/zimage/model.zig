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

const load_parallelism = 2;
const load_dma_chunks = 4;
const load_dma_chunk_size = 16 * zml.MiB;

fn scrubTensorPartitioningToReplicated(model: *Model) void {
    const Ctx = struct {};
    const Visitor = struct {
        fn cb(_: *Ctx, tensor: *zml.Tensor) void {
            tensor._shape = tensor._shape.withReplicatedPartitioning();
        }
    };
    var ctx: Ctx = .{};
    zml.meta.visit(Visitor.cb, &ctx, model);
}

fn scrubTensorStorePartitioningToReplicated(store: *zml.io.TensorStore) void {
    var it = store.registry.tensors.iterator();
    while (it.next()) |entry| {
        entry.value_ptr.shape = entry.value_ptr.shape.withReplicatedPartitioning();
    }
}

fn loadTextEncoderBuffers(
    model: *const text_encoder.TextEncoder,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
) !zml.Bufferized(text_encoder.TextEncoder) {
    return zml.io.load(text_encoder.TextEncoder, model, allocator, io, platform, store, .{
        .dma_chunks = load_dma_chunks,
        .dma_chunk_size = load_dma_chunk_size,
        .progress = progress,
        .shardings = all_shardings,
        .parallelism = load_parallelism,
    });
}

fn loadTransformerBuffers(
    model: *const transformer.Transformer,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
    out: *zml.Bufferized(transformer.Transformer),
) !void {
    for (&out.all_x_embedder, 0..) |*embedder, i| {
        embedder.* = try zml.io.load(zml.nn.Linear, &model.all_x_embedder[i], allocator, io, platform, store, .{
            .dma_chunks = load_dma_chunks,
            .dma_chunk_size = load_dma_chunk_size,
            .progress = progress,
            .shardings = all_shardings,
            .parallelism = load_parallelism,
        });
        errdefer {
            embedder.weight.deinit();
            if (embedder.bias) |*bias| bias.deinit();
        }
    }

    for (&out.all_final_layer, 0..) |*layer, i| {
        layer.* = try zml.io.load(transformer.FinalLayer, &model.all_final_layer[i], allocator, io, platform, store, .{
            .dma_chunks = load_dma_chunks,
            .dma_chunk_size = load_dma_chunk_size,
            .progress = progress,
            .shardings = all_shardings,
            .parallelism = load_parallelism,
        });
        errdefer transformer.FinalLayer.unloadBuffers(layer);
    }

    for (&out.noise_refiner, 0..) |*layer, i| {
        layer.* = try zml.io.load(transformer.ZImageTransformerBlock, &model.noise_refiner[i], allocator, io, platform, store, .{
            .dma_chunks = load_dma_chunks,
            .dma_chunk_size = load_dma_chunk_size,
            .progress = progress,
            .shardings = all_shardings,
            .parallelism = load_parallelism,
        });
        errdefer transformer.ZImageTransformerBlock.unloadBuffers(layer);
    }

    for (&out.context_refiner, 0..) |*layer, i| {
        layer.* = try zml.io.load(transformer.ZImageTransformerBlock, &model.context_refiner[i], allocator, io, platform, store, .{
            .dma_chunks = load_dma_chunks,
            .dma_chunk_size = load_dma_chunk_size,
            .progress = progress,
            .shardings = all_shardings,
            .parallelism = load_parallelism,
        });
        errdefer transformer.ZImageTransformerBlock.unloadBuffers(layer);
    }

    out.t_embedder = try zml.io.load(transformer.TimestepEmbedder, &model.t_embedder, allocator, io, platform, store, .{
        .dma_chunks = load_dma_chunks,
        .dma_chunk_size = load_dma_chunk_size,
        .progress = progress,
        .shardings = all_shardings,
        .parallelism = load_parallelism,
    });
    errdefer transformer.TimestepEmbedder.unloadBuffers(&out.t_embedder);

    out.cap_embedder = try zml.io.load(transformer.Embedder, &model.cap_embedder, allocator, io, platform, store, .{
        .dma_chunks = load_dma_chunks,
        .dma_chunk_size = load_dma_chunk_size,
        .progress = progress,
        .shardings = all_shardings,
        .parallelism = load_parallelism,
    });
    errdefer transformer.Embedder.unloadBuffers(&out.cap_embedder);

    if (model.siglip_embedder) |embedder| {
        out.siglip_embedder = try zml.io.load(transformer.Embedder, &embedder, allocator, io, platform, store, .{
            .dma_chunks = load_dma_chunks,
            .dma_chunk_size = load_dma_chunk_size,
            .progress = progress,
            .shardings = all_shardings,
            .parallelism = load_parallelism,
        });
        errdefer if (out.siglip_embedder) |*loaded| transformer.Embedder.unloadBuffers(loaded);
    } else {
        out.siglip_embedder = null;
    }

    if (model.siglip_refiner) |layers| {
        var loaded_layers: [2]zml.Bufferized(transformer.ZImageTransformerBlock) = undefined;
        for (&loaded_layers, 0..) |*layer, i| {
            layer.* = try zml.io.load(transformer.ZImageTransformerBlock, &layers[i], allocator, io, platform, store, .{
                .dma_chunks = load_dma_chunks,
                .dma_chunk_size = load_dma_chunk_size,
                .progress = progress,
                .shardings = all_shardings,
                .parallelism = load_parallelism,
            });
            errdefer transformer.ZImageTransformerBlock.unloadBuffers(layer);
        }
        out.siglip_refiner = loaded_layers;
        errdefer if (out.siglip_refiner) |*loaded| for (loaded) |*layer| transformer.ZImageTransformerBlock.unloadBuffers(layer);
    } else {
        out.siglip_refiner = null;
    }

    if (model.siglip_pad_token) |token| {
        out.siglip_pad_token = try zml.io.load(zml.Tensor, &token, allocator, io, platform, store, .{
            .dma_chunks = load_dma_chunks,
            .dma_chunk_size = load_dma_chunk_size,
            .progress = progress,
            .shardings = all_shardings,
            .parallelism = load_parallelism,
        });
        errdefer if (out.siglip_pad_token) |*loaded| loaded.deinit();
    } else {
        out.siglip_pad_token = null;
    }

    out.x_pad_token = try zml.io.load(zml.Tensor, &model.x_pad_token, allocator, io, platform, store, .{
        .dma_chunks = load_dma_chunks,
        .dma_chunk_size = load_dma_chunk_size,
        .progress = progress,
        .shardings = all_shardings,
        .parallelism = load_parallelism,
    });
    errdefer out.x_pad_token.deinit();

    out.cap_pad_token = try zml.io.load(zml.Tensor, &model.cap_pad_token, allocator, io, platform, store, .{
        .dma_chunks = load_dma_chunks,
        .dma_chunk_size = load_dma_chunk_size,
        .progress = progress,
        .shardings = all_shardings,
        .parallelism = load_parallelism,
    });
    errdefer out.cap_pad_token.deinit();

    for (&out.layers, 0..) |*layer, i| {
        layer.* = try zml.io.load(transformer.ZImageTransformerBlock, &model.layers[i], allocator, io, platform, store, .{
            .dma_chunks = load_dma_chunks,
            .dma_chunk_size = load_dma_chunk_size,
            .progress = progress,
            .shardings = all_shardings,
            .parallelism = load_parallelism,
        });
        errdefer transformer.ZImageTransformerBlock.unloadBuffers(layer);
    }
}

fn loadVaeConv2dBuffers(
    model: *const vae.Conv2d,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
) !zml.Bufferized(vae.Conv2d) {
    return zml.io.load(vae.Conv2d, model, allocator, io, platform, store, .{
        .dma_chunks = load_dma_chunks,
        .dma_chunk_size = load_dma_chunk_size,
        .progress = progress,
        .shardings = all_shardings,
        .parallelism = load_parallelism,
    });
}

fn loadVaeGroupNormBuffers(
    model: *const vae.GroupNorm,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
) !zml.Bufferized(vae.GroupNorm) {
    return zml.io.load(vae.GroupNorm, model, allocator, io, platform, store, .{
        .dma_chunks = load_dma_chunks,
        .dma_chunk_size = load_dma_chunk_size,
        .progress = progress,
        .shardings = all_shardings,
        .parallelism = load_parallelism,
    });
}

fn loadVaeResnetBlockBuffers(
    model: *const vae.ResnetBlock2D,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
) !zml.Bufferized(vae.ResnetBlock2D) {
    var out: zml.Bufferized(vae.ResnetBlock2D) = undefined;
    out.norm1 = try loadVaeGroupNormBuffers(&model.norm1, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.GroupNorm.unloadBuffers(&out.norm1);
    out.conv1 = try loadVaeConv2dBuffers(&model.conv1, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.Conv2d.unloadBuffers(&out.conv1);
    out.norm2 = try loadVaeGroupNormBuffers(&model.norm2, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.GroupNorm.unloadBuffers(&out.norm2);
    out.conv2 = try loadVaeConv2dBuffers(&model.conv2, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.Conv2d.unloadBuffers(&out.conv2);
    if (model.conv_shortcut) |conv| {
        out.conv_shortcut = try loadVaeConv2dBuffers(&conv, allocator, io, platform, store, progress, all_shardings);
        errdefer if (out.conv_shortcut) |*loaded| vae.Conv2d.unloadBuffers(loaded);
    } else {
        out.conv_shortcut = null;
    }
    return out;
}

fn loadVaeDownsampleBuffers(
    model: *const vae.Downsample2D,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
) !zml.Bufferized(vae.Downsample2D) {
    return .{
        .conv = try loadVaeConv2dBuffers(&model.conv, allocator, io, platform, store, progress, all_shardings),
    };
}

fn loadVaeUpsampleBuffers(
    model: *const vae.Upsample2D,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
) !zml.Bufferized(vae.Upsample2D) {
    return .{
        .conv = try loadVaeConv2dBuffers(&model.conv, allocator, io, platform, store, progress, all_shardings),
    };
}

fn loadVaeAttentionBuffers(
    model: *const vae.Attention2D,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
) !zml.Bufferized(vae.Attention2D) {
    var out: zml.Bufferized(vae.Attention2D) = undefined;
    out.norm = try loadVaeGroupNormBuffers(&model.norm, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.GroupNorm.unloadBuffers(&out.norm);
    out.to_q = try zml.io.load(zml.nn.Linear, &model.to_q, allocator, io, platform, store, .{
        .dma_chunks = load_dma_chunks,
        .dma_chunk_size = load_dma_chunk_size,
        .progress = progress,
        .shardings = all_shardings,
        .parallelism = load_parallelism,
    });
    errdefer {
        out.to_q.weight.deinit();
        if (out.to_q.bias) |*b| b.deinit();
    }
    out.to_k = try zml.io.load(zml.nn.Linear, &model.to_k, allocator, io, platform, store, .{
        .dma_chunks = load_dma_chunks,
        .dma_chunk_size = load_dma_chunk_size,
        .progress = progress,
        .shardings = all_shardings,
        .parallelism = load_parallelism,
    });
    errdefer {
        out.to_k.weight.deinit();
        if (out.to_k.bias) |*b| b.deinit();
    }
    out.to_v = try zml.io.load(zml.nn.Linear, &model.to_v, allocator, io, platform, store, .{
        .dma_chunks = load_dma_chunks,
        .dma_chunk_size = load_dma_chunk_size,
        .progress = progress,
        .shardings = all_shardings,
        .parallelism = load_parallelism,
    });
    errdefer {
        out.to_v.weight.deinit();
        if (out.to_v.bias) |*b| b.deinit();
    }
    out.to_out = try zml.io.load(zml.nn.Linear, &model.to_out, allocator, io, platform, store, .{
        .dma_chunks = load_dma_chunks,
        .dma_chunk_size = load_dma_chunk_size,
        .progress = progress,
        .shardings = all_shardings,
        .parallelism = load_parallelism,
    });
    errdefer {
        out.to_out.weight.deinit();
        if (out.to_out.bias) |*b| b.deinit();
    }
    return out;
}

fn loadVaeMidBlockBuffers(
    model: *const vae.UNetMidBlock2D,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
) !zml.Bufferized(vae.UNetMidBlock2D) {
    var out: zml.Bufferized(vae.UNetMidBlock2D) = undefined;
    for (&out.resnets, 0..) |*resnet, i| {
        resnet.* = try loadVaeResnetBlockBuffers(&model.resnets[i], allocator, io, platform, store, progress, all_shardings);
        errdefer vae.ResnetBlock2D.unloadBuffers(resnet);
    }
    if (model.attention) |attn| {
        out.attention = try loadVaeAttentionBuffers(&attn, allocator, io, platform, store, progress, all_shardings);
        errdefer if (out.attention) |*loaded| vae.Attention2D.unloadBuffers(loaded);
    } else {
        out.attention = null;
    }
    return out;
}

fn loadVaeDownBlockBuffers(
    model: *const vae.DownEncoderBlock2D,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
) !zml.Bufferized(vae.DownEncoderBlock2D) {
    var out: zml.Bufferized(vae.DownEncoderBlock2D) = undefined;
    out.resnets = try allocator.alloc(zml.Bufferized(vae.ResnetBlock2D), model.resnets.len);
    errdefer allocator.free(out.resnets);
    for (out.resnets, 0..) |*resnet, i| {
        resnet.* = try loadVaeResnetBlockBuffers(&model.resnets[i], allocator, io, platform, store, progress, all_shardings);
        errdefer vae.ResnetBlock2D.unloadBuffers(resnet);
    }
    if (model.downsamplers) |downsamplers| {
        out.downsamplers = undefined;
        for (&out.downsamplers.?, 0..) |*downsampler, i| {
            downsampler.* = try loadVaeDownsampleBuffers(&downsamplers[i], allocator, io, platform, store, progress, all_shardings);
            errdefer vae.Downsample2D.unloadBuffers(downsampler);
        }
    } else {
        out.downsamplers = null;
    }
    return out;
}

fn loadVaeUpBlockBuffers(
    model: *const vae.UpDecoderBlock2D,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
) !zml.Bufferized(vae.UpDecoderBlock2D) {
    var out: zml.Bufferized(vae.UpDecoderBlock2D) = undefined;
    out.resnets = try allocator.alloc(zml.Bufferized(vae.ResnetBlock2D), model.resnets.len);
    errdefer allocator.free(out.resnets);
    for (out.resnets, 0..) |*resnet, i| {
        resnet.* = try loadVaeResnetBlockBuffers(&model.resnets[i], allocator, io, platform, store, progress, all_shardings);
        errdefer vae.ResnetBlock2D.unloadBuffers(resnet);
    }
    if (model.upsamplers) |upsamplers| {
        out.upsamplers = undefined;
        for (&out.upsamplers.?, 0..) |*upsampler, i| {
            upsampler.* = try loadVaeUpsampleBuffers(&upsamplers[i], allocator, io, platform, store, progress, all_shardings);
            errdefer vae.Upsample2D.unloadBuffers(upsampler);
        }
    } else {
        out.upsamplers = null;
    }
    return out;
}

fn loadVaeEncoderBuffers(
    model: *const vae.AutoEncoder.Encoder,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
) !zml.Bufferized(vae.AutoEncoder.Encoder) {
    var out: zml.Bufferized(vae.AutoEncoder.Encoder) = undefined;
    out.conv_in = try loadVaeConv2dBuffers(&model.conv_in, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.Conv2d.unloadBuffers(&out.conv_in);
    out.down_blocks = try allocator.alloc(zml.Bufferized(vae.DownEncoderBlock2D), model.down_blocks.len);
    errdefer allocator.free(out.down_blocks);
    for (out.down_blocks, 0..) |*block, i| {
        block.* = try loadVaeDownBlockBuffers(&model.down_blocks[i], allocator, io, platform, store, progress, all_shardings);
        errdefer vae.DownEncoderBlock2D.unloadBuffers(block);
    }
    out.mid_block = try loadVaeMidBlockBuffers(&model.mid_block, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.UNetMidBlock2D.unloadBuffers(&out.mid_block);
    out.conv_norm_out = try loadVaeGroupNormBuffers(&model.conv_norm_out, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.GroupNorm.unloadBuffers(&out.conv_norm_out);
    out.conv_out = try loadVaeConv2dBuffers(&model.conv_out, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.Conv2d.unloadBuffers(&out.conv_out);
    return out;
}

fn loadVaeDecoderBuffers(
    model: *const vae.AutoEncoder.Decoder,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
) !zml.Bufferized(vae.AutoEncoder.Decoder) {
    var out: zml.Bufferized(vae.AutoEncoder.Decoder) = undefined;
    out.conv_in = try loadVaeConv2dBuffers(&model.conv_in, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.Conv2d.unloadBuffers(&out.conv_in);
    out.mid_block = try loadVaeMidBlockBuffers(&model.mid_block, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.UNetMidBlock2D.unloadBuffers(&out.mid_block);
    out.up_blocks = try allocator.alloc(zml.Bufferized(vae.UpDecoderBlock2D), model.up_blocks.len);
    errdefer allocator.free(out.up_blocks);
    for (out.up_blocks, 0..) |*block, i| {
        block.* = try loadVaeUpBlockBuffers(&model.up_blocks[i], allocator, io, platform, store, progress, all_shardings);
        errdefer vae.UpDecoderBlock2D.unloadBuffers(block);
    }
    out.conv_norm_out = try loadVaeGroupNormBuffers(&model.conv_norm_out, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.GroupNorm.unloadBuffers(&out.conv_norm_out);
    out.conv_out = try loadVaeConv2dBuffers(&model.conv_out, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.Conv2d.unloadBuffers(&out.conv_out);
    return out;
}

fn loadVaeBuffers(
    model: *const vae.AutoEncoder,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    all_shardings: []const zml.Sharding,
) !zml.Bufferized(vae.AutoEncoder) {
    var out: zml.Bufferized(vae.AutoEncoder) = undefined;
    out.encoder = try loadVaeEncoderBuffers(&model.encoder, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.AutoEncoder.Encoder.unloadBuffers(&out.encoder);
    out.decoder = try loadVaeDecoderBuffers(&model.decoder, allocator, io, platform, store, progress, all_shardings);
    errdefer vae.AutoEncoder.Decoder.unloadBuffers(&out.decoder);
    if (model.quant_conv) |conv| {
        out.quant_conv = try loadVaeConv2dBuffers(&conv, allocator, io, platform, store, progress, all_shardings);
        errdefer if (out.quant_conv) |*loaded| vae.Conv2d.unloadBuffers(loaded);
    } else {
        out.quant_conv = null;
    }
    if (model.post_quant_conv) |conv| {
        out.post_quant_conv = try loadVaeConv2dBuffers(&conv, allocator, io, platform, store, progress, all_shardings);
        errdefer if (out.post_quant_conv) |*loaded| vae.Conv2d.unloadBuffers(loaded);
    } else {
        out.post_quant_conv = null;
    }
    return out;
}

fn unloadVaeDownBlockBuffers(model: *zml.Bufferized(vae.DownEncoderBlock2D), allocator: std.mem.Allocator) void {
    for (model.resnets) |*resnet| {
        vae.ResnetBlock2D.unloadBuffers(resnet);
    }
    allocator.free(model.resnets);

    if (model.downsamplers) |*downsamplers| {
        for (downsamplers) |*downsampler| {
            vae.Downsample2D.unloadBuffers(downsampler);
        }
    }
}

fn unloadVaeUpBlockBuffers(model: *zml.Bufferized(vae.UpDecoderBlock2D), allocator: std.mem.Allocator) void {
    for (model.resnets) |*resnet| {
        vae.ResnetBlock2D.unloadBuffers(resnet);
    }
    allocator.free(model.resnets);

    if (model.upsamplers) |*upsamplers| {
        for (upsamplers) |*upsampler| {
            vae.Upsample2D.unloadBuffers(upsampler);
        }
    }
}

fn unloadVaeEncoderBuffers(model: *zml.Bufferized(vae.AutoEncoder.Encoder), allocator: std.mem.Allocator) void {
    vae.Conv2d.unloadBuffers(&model.conv_in);
    for (model.down_blocks) |*block| {
        unloadVaeDownBlockBuffers(block, allocator);
    }
    allocator.free(model.down_blocks);
    vae.UNetMidBlock2D.unloadBuffers(&model.mid_block);
    vae.GroupNorm.unloadBuffers(&model.conv_norm_out);
    vae.Conv2d.unloadBuffers(&model.conv_out);
}

fn unloadVaeDecoderBuffers(model: *zml.Bufferized(vae.AutoEncoder.Decoder), allocator: std.mem.Allocator) void {
    vae.Conv2d.unloadBuffers(&model.conv_in);
    vae.UNetMidBlock2D.unloadBuffers(&model.mid_block);
    for (model.up_blocks) |*block| {
        unloadVaeUpBlockBuffers(block, allocator);
    }
    allocator.free(model.up_blocks);
    vae.GroupNorm.unloadBuffers(&model.conv_norm_out);
    vae.Conv2d.unloadBuffers(&model.conv_out);
}

fn unloadVaeBuffers(model: *zml.Bufferized(vae.AutoEncoder), allocator: std.mem.Allocator) void {
    unloadVaeEncoderBuffers(&model.encoder, allocator);
    unloadVaeDecoderBuffers(&model.decoder, allocator);
    if (model.quant_conv) |*conv| {
        vae.Conv2d.unloadBuffers(conv);
    }
    if (model.post_quant_conv) |*conv| {
        vae.Conv2d.unloadBuffers(conv);
    }
}

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
        self.text_encoder = try text_encoder.TextEncoder.init(allocator, store.withPrefix("text_encoder"), config.text_encoder);
        errdefer self.text_encoder.deinit(allocator);

        log.info("Initializing transformer model graph...", .{});
        try self.transformer.init(allocator, store.withPrefix("transformer"), config.transformer);
        errdefer self.transformer.deinit(allocator);

        log.info("Initializing VAE model graph...", .{});
        self.vae = try vae.AutoEncoder.init(store.withPrefix("vae"), config.vae);
        errdefer self.vae.deinit(allocator);
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.text_encoder.deinit(allocator);
        self.transformer.deinit(allocator);
        self.vae.deinit(allocator);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        text_encoder.Qwen3Model.unloadBuffers(&self.text_encoder.inner.model, allocator);
        if (self.text_encoder.inner.lm_head) |*lm_head| lm_head.weight.deinit();
        transformer.Transformer.unloadBuffers(&self.transformer);
        unloadVaeBuffers(&self.vae, allocator);
    }
};

pub const SchedulerStepKernel = struct {
    pub fn step(
        _: SchedulerStepKernel,
        model_output: zml.Tensor,
        sample: zml.Tensor,
        current_sigma: zml.Tensor,
        next_sigma: zml.Tensor,
    ) zml.Tensor {
        const sample_f32 = sample.convert(.f32);
        const dt = next_sigma.convert(.f32)
            .sub(current_sigma.convert(.f32))
            .broad(sample_f32.shape());

        return sample_f32
            .add(dt.mul(model_output.convert(.f32)))
            .convert(model_output.dtype());
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
        out: *Buffers,
    ) !void {
        try loadBuffersImpl(self, allocator, io, platform, store, progress, shardings, out);
    }

    fn loadBuffersImpl(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
        out: *Buffers,
    ) !void {
        scrubTensorPartitioningToReplicated(&@constCast(self).inner);
        scrubTensorStorePartitioningToReplicated(store);
        progress.increaseEstimatedTotalItems(store.view().count());
        _ = shardings;
        const all_shardings = [_]zml.Sharding{platform.replicated_sharding};

        out.text_encoder = try loadTextEncoderBuffers(&self.inner.text_encoder, allocator, io, platform, store, progress, &all_shardings);
        errdefer text_encoder.Qwen3Model.unloadBuffers(&out.text_encoder.inner.model, allocator);
        errdefer if (out.text_encoder.inner.lm_head) |*lm_head| lm_head.weight.deinit();

        try loadTransformerBuffers(&self.inner.transformer, allocator, io, platform, store, progress, &all_shardings, &out.transformer);
        errdefer transformer.Transformer.unloadBuffers(&out.transformer);

        out.vae = try loadVaeBuffers(&self.inner.vae, allocator, io, platform, store, progress, &all_shardings);
        errdefer unloadVaeBuffers(&out.vae, allocator);
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
        height: u32,
        width: u32,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        _ = backend;
        if (height % 16 != 0 or width % 16 != 0) return inference.InferenceErrors.InvalidImageSize;
        const params = inference.CompilationParameters.init(
            @intCast(seqlen),
            1,
            height / 8,
            width / 8,
            shardings,
        );
        return inference.CompiledModel.init(allocator, io, platform, self, params, progress);
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
