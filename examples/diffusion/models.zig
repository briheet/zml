const std = @import("std");
const zml = @import("zml");

pub const zimageModel = @import("models/zimage.zig");
pub const pipeline = zimageModel.inference.InferencePipeline;
pub const InferenceErrors = zimageModel.inference.InferenceErrors;
pub const common = @import("models/common.zig");
pub const Shardings = common.Shardings;

const log = std.log.scoped(.diffusion);

pub const ModelType = enum {
    zimage,
};

const RawConfig = struct {
    _class_name: []const u8,
};

pub const LoadedModel = union(ModelType) {
    zimage: *zimageModel.LoadedModel,

    pub fn load(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
    ) !LoadedModel {
        const model_type = try detectModelType(allocator, io, repo);
        log.info("Detected mode type: {}", .{model_type});
        return switch (model_type) {
            .zimage => blk: {
                const model = try allocator.create(zimageModel.LoadedModel);
                errdefer allocator.destroy(model);
                model.* = try zimageModel.LoadedModel.init(allocator, io, repo, store);
                break :blk .{ .zimage = model };
            },
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .zimage => |m| {
                m.deinit(allocator);
                allocator.destroy(m);
            },
        }
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.attention.Backend,
        shardings: Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const inner: CompiledModel.Inner = switch (self.*) {
            .zimage => |m| .{ .zimage = try m.compile(
                allocator,
                io,
                platform,
                backend,
                shardings,
                seqlen,
                progress,
            ) },
        };

        return .{
            .inner = inner,
            .seqlen = @intCast(seqlen),
        };
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: Shardings,
    ) !Buffers {
        return switch (self.*) {
            .zimage => |m| .{ .zimage = try m.loadBuffers(allocator, io, platform, store, progress, shardings) },
        };
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .zimage => |loaded_model| switch (buffers.*) {
                .zimage => |*loaded_buffers| loaded_model.unloadBuffers(loaded_buffers, allocator),
            },
        }
    }
};

pub const CompiledModel = struct {
    const Inner = union(ModelType) {
        zimage: zimageModel.inference.CompiledModel,
    };

    inner: Inner,
    seqlen: u32,

    pub fn deinit(self: *CompiledModel) void {
        switch (self.inner) {
            inline else => |*m| m.deinit(),
        }
    }
};

pub const Buffers = union(ModelType) {
    zimage: zimageModel.Buffers,
};

pub fn detectModelType(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir) !ModelType {
    const file = try repo.openFile(io, "model_index.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();
    const parsed = try std.json.parseFromTokenSource(RawConfig, allocator, &reader, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    if (std.mem.eql(u8, parsed.value._class_name, "ZImagePipeline")) return .zimage;
    return error.UnknownModelType;
}
