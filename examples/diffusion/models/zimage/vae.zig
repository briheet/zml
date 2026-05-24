const std = @import("std");

const zml = @import("zml");

// ZImagePipeline uses AutoEncoderKL (Autoencoder with KL loss)
pub const AutoEncoder = struct {
    // The configs defined here depend on the models vae/config.json
    // One can find it here := https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/vae/config.json
    pub const Config = struct {
        act_fn: []const u8 = "silu",
        block_out_channels: [4]u32 = .{ 128, 256, 512, 512 },
        down_block_types: [4][]const u8 = .{
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        },
        force_upcast: bool = false,
        in_channels: u32 = 3,
        latent_channels: u32 = 16,
        latents_mean: ?f32 = null,
        latents_std: ?f32 = null,
        layers_per_block: u32 = 2,
        mid_block_add_attention: bool = true,
        norm_num_groups: u32 = 32,
        out_channels: u32 = 3,
        sample_size: u32 = 1024,
        scaling_factor: f32 = 0.3611,
        shift_factor: f32 = 0.1159,
        up_block_types: [4][]const u8 = .{
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        },
        use_post_quant_conv: bool = false,
        use_quant_conv: bool = false,
    };

    pub const AutoencoderKLOutput = struct {
        latent: zml.Tensor,
    };

    pub const DecoderOutput = struct {
        sample: zml.Tensor,
    };

    config: Config,
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: ?Conv2d,
    post_quant_conv: ?Conv2d,
    use_slicing: bool,
    use_tiling: bool,
    tile_sample_min_size: u32,
    tile_latent_min_size: u32,
    tile_overlap_factor: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) !AutoEncoder {
        return .{
            .config = config,
            .encoder = try Encoder.init(store.withPrefix("encoder"), config),
            .decoder = try Decoder.init(store.withPrefix("decoder"), config),
            .quant_conv = if (config.use_quant_conv)
                Conv2d.init(
                    store.withPrefix("quant_conv"),
                    2 * config.latent_channels,
                    2 * config.latent_channels,
                    1,
                    1,
                    0,
                )
            else
                null,
            .post_quant_conv = if (config.use_post_quant_conv)
                Conv2d.init(
                    store.withPrefix("post_quant_conv"),
                    config.latent_channels,
                    config.latent_channels,
                    1,
                    1,
                    0,
                )
            else
                null,
            .use_slicing = false,
            .use_tiling = false,
            .tile_sample_min_size = config.sample_size,
            .tile_latent_min_size = @divExact(config.sample_size, 1 << @as(std.math.Log2Int(u32), @intCast(config.block_out_channels.len - 1))),
            .tile_overlap_factor = 0.25,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AutoEncoder)) void {
        Encoder.unloadBuffers(&self.encoder);
        Decoder.unloadBuffers(&self.decoder);
        if (self.quant_conv) |*conv| Conv2d.unloadBuffers(conv);
        if (self.post_quant_conv) |*conv| Conv2d.unloadBuffers(conv);
    }

    pub fn deinit(self: AutoEncoder, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.encoder.deinit();
        self.decoder.deinit();
    }

    fn encodeInternal(self: AutoEncoder, x: zml.Tensor) zml.Tensor {
        var enc = self.encoder.forward(x);
        if (self.quant_conv) |conv| {
            enc = conv.forward(enc);
        }
        return enc;
    }

    pub fn encode(self: AutoEncoder, x: zml.Tensor) AutoencoderKLOutput {
        return .{
            .latent = self.encodeInternal(x),
        };
    }

    pub fn decodeInternal(self: AutoEncoder, z: zml.Tensor) zml.Tensor {
        var latents = z;
        if (self.post_quant_conv) |conv| {
            latents = conv.forward(latents);
        }
        return self.decoder.forward(latents, null);
    }

    pub fn decode(self: AutoEncoder, z: zml.Tensor) DecoderOutput {
        return .{
            .sample = self.decodeInternal(z),
        };
    }

    // Encoder layer for variational autoencoder that encoder input data into latent representation
    // Can be used in other vae's aswell
    pub const Encoder = struct {
        layers_per_block: u32,
        conv_in: Conv2d,
        down_blocks: []DownEncoderBlock2D,
        mid_block: UNetMidBlock2D,
        conv_norm_out: GroupNorm,
        conv_out: Conv2d,
        act_fn: []const u8,
        gradient_checkpointing: bool,

        pub fn init(store: zml.io.TensorStore.View, config: Config) !Encoder {
            const down_blocks = try std.heap.page_allocator.alloc(DownEncoderBlock2D, config.block_out_channels.len);
            errdefer std.heap.page_allocator.free(down_blocks);

            var output_channel = config.block_out_channels[0];
            for (config.down_block_types, config.block_out_channels, 0..) |down_block_type, block_out, i| {
                const input_channel = output_channel;
                output_channel = block_out;
                const is_final_block = i == config.block_out_channels.len - 1;
                down_blocks[i] = try DownEncoderBlock2D.init(
                    store.withPrefix("down_blocks").withLayer(i),
                    down_block_type,
                    config.layers_per_block,
                    input_channel,
                    output_channel,
                    !is_final_block,
                    config.act_fn,
                    config.norm_num_groups,
                );
            }

            return .{
                .layers_per_block = config.layers_per_block,
                .conv_in = Conv2d.init(store.withPrefix("conv_in"), config.in_channels, config.block_out_channels[0], 3, 1, 1),
                .down_blocks = down_blocks,
                .mid_block = try UNetMidBlock2D.init(
                    store.withPrefix("mid_block"),
                    config.block_out_channels[config.block_out_channels.len - 1],
                    config.act_fn,
                    config.norm_num_groups,
                    config.mid_block_add_attention,
                ),
                .conv_norm_out = GroupNorm.init(
                    store.withPrefix("conv_norm_out"),
                    config.block_out_channels[config.block_out_channels.len - 1],
                    config.norm_num_groups,
                    1e-6,
                ),
                .conv_out = Conv2d.init(
                    store.withPrefix("conv_out"),
                    config.block_out_channels[config.block_out_channels.len - 1],
                    2 * config.latent_channels,
                    3,
                    1,
                    1,
                ),
                .act_fn = config.act_fn,
                .gradient_checkpointing = false,
            };
        }

        pub fn deinit(self: Encoder) void {
            std.heap.page_allocator.free(self.down_blocks);
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Encoder)) void {
            Conv2d.unloadBuffers(&self.conv_in);
            for (self.down_blocks) |*block| DownEncoderBlock2D.unloadBuffers(block);
            UNetMidBlock2D.unloadBuffers(&self.mid_block);
            GroupNorm.unloadBuffers(&self.conv_norm_out);
            Conv2d.unloadBuffers(&self.conv_out);
        }

        pub fn forward(self: Encoder, sample: zml.Tensor) zml.Tensor {
            var out = self.conv_in.forward(sample);
            for (self.down_blocks) |block| {
                out = block.forward(out);
            }
            out = self.mid_block.forward(out);
            out = self.conv_norm_out.forward(out);
            out = applyAct(self.act_fn, out);
            return self.conv_out.forward(out);
        }
    };

    // Decoder layer for variational autoencoder
    pub const Decoder = struct {
        layers_per_block: u32,
        conv_in: Conv2d,
        mid_block: UNetMidBlock2D,
        up_blocks: []UpDecoderBlock2D,
        conv_norm_out: GroupNorm,
        conv_out: Conv2d,
        act_fn: []const u8,
        gradient_checkpointing: bool,

        pub fn init(store: zml.io.TensorStore.View, config: Config) !Decoder {
            const up_blocks = try std.heap.page_allocator.alloc(UpDecoderBlock2D, config.block_out_channels.len);
            errdefer std.heap.page_allocator.free(up_blocks);

            const reversed = [_]u32{
                config.block_out_channels[3],
                config.block_out_channels[2],
                config.block_out_channels[1],
                config.block_out_channels[0],
            };
            var output_channel = reversed[0];

            for (config.up_block_types, reversed, 0..) |up_block_type, block_out, i| {
                const prev_output_channel = output_channel;
                output_channel = block_out;
                const is_final_block = i == reversed.len - 1;
                up_blocks[i] = try UpDecoderBlock2D.init(
                    store.withPrefix("up_blocks").withLayer(i),
                    up_block_type,
                    config.layers_per_block + 1,
                    prev_output_channel,
                    output_channel,
                    !is_final_block,
                    config.act_fn,
                    config.norm_num_groups,
                );
            }

            return .{
                .layers_per_block = config.layers_per_block,
                .conv_in = Conv2d.init(
                    store.withPrefix("conv_in"),
                    config.latent_channels,
                    config.block_out_channels[config.block_out_channels.len - 1],
                    3,
                    1,
                    1,
                ),
                .mid_block = try UNetMidBlock2D.init(
                    store.withPrefix("mid_block"),
                    config.block_out_channels[config.block_out_channels.len - 1],
                    config.act_fn,
                    config.norm_num_groups,
                    config.mid_block_add_attention,
                ),
                .up_blocks = up_blocks,
                .conv_norm_out = GroupNorm.init(
                    store.withPrefix("conv_norm_out"),
                    config.block_out_channels[0],
                    config.norm_num_groups,
                    1e-6,
                ),
                .conv_out = Conv2d.init(
                    store.withPrefix("conv_out"),
                    config.block_out_channels[0],
                    config.out_channels,
                    3,
                    1,
                    1,
                ),
                .act_fn = config.act_fn,
                .gradient_checkpointing = false,
            };
        }

        pub fn deinit(self: Decoder) void {
            std.heap.page_allocator.free(self.up_blocks);
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Decoder)) void {
            Conv2d.unloadBuffers(&self.conv_in);
            UNetMidBlock2D.unloadBuffers(&self.mid_block);
            for (self.up_blocks) |*block| UpDecoderBlock2D.unloadBuffers(block);
            GroupNorm.unloadBuffers(&self.conv_norm_out);
            Conv2d.unloadBuffers(&self.conv_out);
        }

        pub fn forward(self: Decoder, sample: zml.Tensor, latent_embeds: ?zml.Tensor) zml.Tensor {
            _ = latent_embeds;
            var out = self.conv_in.forward(sample);
            out = self.mid_block.forward(out);
            for (self.up_blocks) |block| {
                out = block.forward(out);
            }
            out = self.conv_norm_out.forward(out);
            out = applyAct(self.act_fn, out);
            return self.conv_out.forward(out);
        }
    };
};

fn applyAct(act_fn: []const u8, x: zml.Tensor) zml.Tensor {
    if (std.mem.eql(u8, act_fn, "silu")) return x.silu();
    @panic("unsupported VAE activation");
}

pub const Conv2d = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor,
    stride: u32,
    padding: u32,

    pub fn init(
        store: zml.io.TensorStore.View,
        in_channels: u32,
        out_channels: u32,
        kernel_size: u32,
        stride: u32,
        padding: u32,
    ) Conv2d {
        _ = in_channels;
        _ = out_channels;
        _ = kernel_size;
        return .{
            .weight = store.createTensor("weight", .{ .cout, .cin, .kh, .kw }, .{
                .cout = .model,
                .cin = .replicated,
                .kh = .replicated,
                .kw = .replicated,
            }),
            .bias = store.maybeCreateTensor("bias", .{.cout}, .{ .cout = .model }),
            .stride = stride,
            .padding = padding,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Conv2d)) void {
        self.weight.deinit();
        if (self.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Conv2d, input: zml.Tensor) zml.Tensor {
        var out = zml.Tensor.conv2d(input, self.weight, .{
            .window_strides = &.{ self.stride, self.stride },
            .padding = &.{ self.padding, self.padding, self.padding, self.padding },
            .input_batch_dimension = input.axis(.b),
            .input_feature_dimension = input.axis(.c),
            .input_spatial_dimensions = &.{ input.axis(.h), input.axis(.w) },
            .kernel_output_feature_dimension = self.weight.axis(.cout),
            .kernel_input_feature_dimension = self.weight.axis(.cin),
            .kernel_spatial_dimensions = &.{ self.weight.axis(.kh), self.weight.axis(.kw) },
            .output_batch_dimension = input.axis(.b),
            .output_feature_dimension = input.axis(.c),
            .output_spatial_dimensions = &.{ input.axis(.h), input.axis(.w) },
        }).rename(.{ .cout = .c });

        if (self.bias) |bias| {
            out = out.add(bias.rename(.{ .cout = .c }).broad(out.shape()));
        }
        return out;
    }
};

pub const GroupNorm = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor,
    num_groups: u32,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, num_channels: u32, num_groups: u32, eps: f32) GroupNorm {
        _ = num_channels;
        return .{
            .weight = store.createTensor("weight", .{.c}, .{ .c = .replicated }),
            .bias = store.maybeCreateTensor("bias", .{.c}, .{ .c = .replicated }),
            .num_groups = num_groups,
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GroupNorm)) void {
        self.weight.deinit();
        if (self.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: GroupNorm, x: zml.Tensor) zml.Tensor {
        const split = x.splitAxis(.c, .{ .g = self.num_groups, .cg = .auto });
        const grouped = split.merge(.{ .d = .{ .cg, .h, .w } });
        const normed = zml.nn.normalizeVariance(grouped, self.eps)
            .splitAxis(.d, .{ .cg = @divExact(@as(u32, @intCast(x.dim(.c))), self.num_groups), .hw = x.dim(.h) * x.dim(.w) })
            .splitAxis(.hw, .{ .h = x.dim(.h), .w = x.dim(.w) })
            .merge(.{ .c = .{ .g, .cg } });
        var out = normed.mul(self.weight.broad(normed.shape()));
        if (self.bias) |bias| out = out.add(bias.broad(out.shape()));
        return out;
    }
};

pub const ResnetBlock2D = struct {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: ?Conv2d,
    act_fn: []const u8,
    output_scale_factor: f32,

    pub fn init(
        store: zml.io.TensorStore.View,
        in_channels: u32,
        out_channels: u32,
        groups: u32,
        act_fn: []const u8,
    ) ResnetBlock2D {
        return .{
            .norm1 = GroupNorm.init(store.withPrefix("norm1"), in_channels, groups, 1e-6),
            .conv1 = Conv2d.init(store.withPrefix("conv1"), in_channels, out_channels, 3, 1, 1),
            .norm2 = GroupNorm.init(store.withPrefix("norm2"), out_channels, groups, 1e-6),
            .conv2 = Conv2d.init(store.withPrefix("conv2"), out_channels, out_channels, 3, 1, 1),
            .conv_shortcut = if (in_channels != out_channels)
                Conv2d.init(store.withPrefix("conv_shortcut"), in_channels, out_channels, 1, 1, 0)
            else
                null,
            .act_fn = act_fn,
            .output_scale_factor = 1.0,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ResnetBlock2D)) void {
        GroupNorm.unloadBuffers(&self.norm1);
        Conv2d.unloadBuffers(&self.conv1);
        GroupNorm.unloadBuffers(&self.norm2);
        Conv2d.unloadBuffers(&self.conv2);
        if (self.conv_shortcut) |*conv| Conv2d.unloadBuffers(conv);
    }

    pub fn forward(self: ResnetBlock2D, input: zml.Tensor) zml.Tensor {
        var hidden = self.norm1.forward(input);
        hidden = applyAct(self.act_fn, hidden);
        hidden = self.conv1.forward(hidden);
        hidden = self.norm2.forward(hidden);
        hidden = applyAct(self.act_fn, hidden);
        hidden = self.conv2.forward(hidden);

        const shortcut = if (self.conv_shortcut) |conv| conv.forward(input) else input;
        return shortcut.add(hidden).scale(1.0 / self.output_scale_factor);
    }
};

pub const Downsample2D = struct {
    conv: Conv2d,

    pub fn init(store: zml.io.TensorStore.View, channels: u32, out_channels: u32, padding: u32) Downsample2D {
        return .{
            .conv = Conv2d.init(store.withPrefix("conv"), channels, out_channels, 3, 2, padding),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Downsample2D)) void {
        Conv2d.unloadBuffers(&self.conv);
    }

    pub fn forward(self: Downsample2D, x: zml.Tensor) zml.Tensor {
        return self.conv.forward(x);
    }
};

pub const Upsample2D = struct {
    conv: Conv2d,

    pub fn init(store: zml.io.TensorStore.View, channels: u32, out_channels: u32) Upsample2D {
        return .{
            .conv = Conv2d.init(store.withPrefix("conv"), channels, out_channels, 3, 1, 1),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Upsample2D)) void {
        Conv2d.unloadBuffers(&self.conv);
    }

    pub fn forward(self: Upsample2D, x: zml.Tensor) zml.Tensor {
        const upsampled = zml.nn.upsample(x, .{ .scale_factor = &.{ 2, 2 }, .mode = .nearest });
        return self.conv.forward(upsampled);
    }
};

pub const DownEncoderBlock2D = struct {
    resnets: []ResnetBlock2D,
    downsamplers: ?[1]Downsample2D,

    pub fn init(
        store: zml.io.TensorStore.View,
        block_type: []const u8,
        num_layers: u32,
        in_channels: u32,
        out_channels: u32,
        add_downsample: bool,
        act_fn: []const u8,
        norm_num_groups: u32,
    ) !DownEncoderBlock2D {
        _ = block_type;
        const resnets = try std.heap.page_allocator.alloc(ResnetBlock2D, num_layers);
        errdefer std.heap.page_allocator.free(resnets);

        for (resnets, 0..) |*resnet, i| {
            const resnet_in_channels = if (i == 0) in_channels else out_channels;
            resnet.* = ResnetBlock2D.init(
                store.withPrefix("resnets").withLayer(i),
                resnet_in_channels,
                out_channels,
                norm_num_groups,
                act_fn,
            );
        }

        return .{
            .resnets = resnets,
            .downsamplers = if (add_downsample)
                .{Downsample2D.init(store.withPrefix("downsamplers").withLayer(0), out_channels, out_channels, 0)}
            else
                null,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DownEncoderBlock2D)) void {
        for (self.resnets) |*resnet| ResnetBlock2D.unloadBuffers(resnet);
        if (self.downsamplers) |*downsamplers| {
            for (downsamplers) |*downsampler| Downsample2D.unloadBuffers(downsampler);
        }
    }

    pub fn forward(self: DownEncoderBlock2D, x: zml.Tensor) zml.Tensor {
        var hidden = x;
        for (self.resnets) |resnet| {
            hidden = resnet.forward(hidden);
        }

        if (self.downsamplers) |downsamplers| {
            for (downsamplers) |downsampler| hidden = downsampler.forward(hidden);
        }
        return hidden;
    }
};

pub const UpDecoderBlock2D = struct {
    resnets: []ResnetBlock2D,
    upsamplers: ?[1]Upsample2D,

    pub fn init(
        store: zml.io.TensorStore.View,
        block_type: []const u8,
        num_layers: u32,
        in_channels: u32,
        out_channels: u32,
        add_upsample: bool,
        act_fn: []const u8,
        norm_num_groups: u32,
    ) !UpDecoderBlock2D {
        _ = block_type;
        const resnets = try std.heap.page_allocator.alloc(ResnetBlock2D, num_layers);
        errdefer std.heap.page_allocator.free(resnets);

        for (resnets, 0..) |*resnet, i| {
            const resnet_in_channels = if (i == 0) in_channels else out_channels;
            resnet.* = ResnetBlock2D.init(
                store.withPrefix("resnets").withLayer(i),
                resnet_in_channels,
                out_channels,
                norm_num_groups,
                act_fn,
            );
        }

        return .{
            .resnets = resnets,
            .upsamplers = if (add_upsample)
                .{Upsample2D.init(store.withPrefix("upsamplers").withLayer(0), out_channels, out_channels)}
            else
                null,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(UpDecoderBlock2D)) void {
        for (self.resnets) |*resnet| ResnetBlock2D.unloadBuffers(resnet);
        if (self.upsamplers) |*upsamplers| {
            for (upsamplers) |*upsampler| Upsample2D.unloadBuffers(upsampler);
        }
    }

    pub fn forward(self: UpDecoderBlock2D, x: zml.Tensor) zml.Tensor {
        var hidden = x;
        for (self.resnets) |resnet| {
            hidden = resnet.forward(hidden);
        }

        if (self.upsamplers) |upsamplers| {
            for (upsamplers) |upsampler| hidden = upsampler.forward(hidden);
        }
        return hidden;
    }
};

pub const UNetMidBlock2D = struct {
    resnets: [2]ResnetBlock2D,
    attention: ?Attention2D,

    pub fn init(
        store: zml.io.TensorStore.View,
        in_channels: u32,
        act_fn: []const u8,
        norm_num_groups: u32,
        add_attention: bool,
    ) !UNetMidBlock2D {
        return .{
            .resnets = .{
                ResnetBlock2D.init(store.withPrefix("resnets").withLayer(0), in_channels, in_channels, norm_num_groups, act_fn),
                ResnetBlock2D.init(store.withPrefix("resnets").withLayer(1), in_channels, in_channels, norm_num_groups, act_fn),
            },
            .attention = if (add_attention)
                Attention2D.init(store.withPrefix("attentions").withLayer(0), in_channels, @max(@divExact(in_channels, in_channels), 1), in_channels)
            else
                null,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(UNetMidBlock2D)) void {
        for (&self.resnets) |*resnet| ResnetBlock2D.unloadBuffers(resnet);
        if (self.attention) |*attn| Attention2D.unloadBuffers(attn);
    }

    pub fn forward(self: UNetMidBlock2D, x: zml.Tensor) zml.Tensor {
        var hidden = self.resnets[0].forward(x);
        if (self.attention) |attn| hidden = attn.forward(hidden);
        hidden = self.resnets[1].forward(hidden);
        return hidden;
    }
};

pub const Attention2D = struct {
    norm: GroupNorm,
    to_q: zml.nn.Linear,
    to_k: zml.nn.Linear,
    to_v: zml.nn.Linear,
    to_out: zml.nn.Linear,
    heads: u32,
    dim_head: u32,

    pub fn init(store: zml.io.TensorStore.View, channels: u32, heads: u32, dim_head: u32) Attention2D {
        return .{
            .norm = GroupNorm.init(store.withPrefix("group_norm"), channels, 32, 1e-6),
            .to_q = .init(store.withPrefix("to_q").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }), store.withPrefix("to_q").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }), .d),
            .to_k = .init(store.withPrefix("to_k").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }), store.withPrefix("to_k").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }), .d),
            .to_v = .init(store.withPrefix("to_v").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }), store.withPrefix("to_v").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }), .d),
            .to_out = .init(store.withPrefix("to_out").withLayer(0).createTensor("weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .model }), store.withPrefix("to_out").withLayer(0).maybeCreateTensor("bias", .{.d}, .{ .d = .replicated }), .dout),
            .heads = heads,
            .dim_head = dim_head,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention2D)) void {
        GroupNorm.unloadBuffers(&self.norm);
        self.to_q.weight.deinit();
        if (self.to_q.bias) |*bias| bias.deinit();
        self.to_k.weight.deinit();
        if (self.to_k.bias) |*bias| bias.deinit();
        self.to_v.weight.deinit();
        if (self.to_v.bias) |*bias| bias.deinit();
        self.to_out.weight.deinit();
        if (self.to_out.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Attention2D, x: zml.Tensor) zml.Tensor {
        const residual = x;
        const normed = self.norm.forward(x);
        const b = normed.dim(.b);
        const h = normed.dim(.h);
        const w = normed.dim(.w);
        const flat = normed.transpose(.{ .b, .h, .w, .c }).reshape(.{ .b = b, .s = h * w, .d = normed.dim(.c) });

        const q = self.to_q.forward(flat).splitAxis(.dout, .{ .h = self.heads, .hd = self.dim_head }).rename(.{ .s = .q });
        const k = self.to_k.forward(flat).splitAxis(.dout, .{ .h = self.heads, .hd = self.dim_head }).rename(.{ .s = .k });
        const v = self.to_v.forward(flat).splitAxis(.dout, .{ .h = self.heads, .hd = self.dim_head }).rename(.{ .s = .k });
        const attn = zml.nn.sdpa(q.squeeze(.b), k.squeeze(.b), v.squeeze(.b), .{})
            .insertAxes(.q, .{.b})
            .rename(.{ .q = .s })
            .merge(.{ .d = .{ .h, .hd } });
        const out = self.to_out.forward(attn).reshape(.{ .b = b, .h = h, .w = w, .c = residual.dim(.c) }).transpose(.{ .b, .c, .h, .w });
        return residual.add(out);
    }
};
