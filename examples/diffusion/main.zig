const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const models = @import("models.zig");

const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.diffusion);

// comptime {
//     _ = models;
// }

const Args = struct {
    model: []const u8,
    prompt: ?[]const u8 = null,
    seqlen: u32 = 512,
    steps: u32 = 50,
    output: []const u8 = "zimage.ppm",
    seed: u64 = 0,
};

const GenerationOptions = struct {
    steps: u32,
    output: []const u8,
    seed: u64,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // `bazel run` executes binaries from Bazel's runfiles tree by default.
    // If available, switch back to the shell's original working directory.
    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = stdx.flags.parse(init.minimal.args, Args);

    //
    // Virtual File system
    //
    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    defer http_client.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var s3_vfs: zml.io.VFS.S3 = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer s3_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("hf", hf_vfs.io());
    try vfs.register("s3", s3_vfs.io());

    const io = vfs.io();

    //
    // Platform and backend selection
    //
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    log.info("\n{f}", .{platform.fmtVerbose()});

    //
    // Model initalization
    //
    log.info("Resolving model repository..", .{});
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    log.info("Initializing model..", .{});
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var model = try models.LoadedModel.load(allocator, io, repo, store.view());
    defer model.deinit(allocator);

    const shardings: models.Shardings = try .init(platform);

    //
    // Load the model and compile it
    //
    var progress = std.Progress.start(io, .{ .root_name = args.model });
    errdefer progress.end();

    var inference_pipeline = switch (model) {
        .zimage => |*zimage_loaded_model| try models.pipeline.init(
            allocator,
            io,
            repo,
            platform,
            zimage_loaded_model,
            &store,
            shardings,
            args.seqlen,
            &progress,
        ),
    };
    defer inference_pipeline.deinit(allocator);

    progress.end();
}

pub fn printZmlLogo(io: std.Io) !void {
    const LOGO =
        \\
        \\
        \\ ███████╗███╗   ███╗██╗
        \\ ╚══███╔╝████╗ ████║██║
        \\   ███╔╝ ██╔████╔██║██║
        \\  ███╔╝  ██║╚██╔╝██║██║  .ai
        \\ ███████╗██║ ╚═╝ ██║███████╗
        \\ ╚══════╝╚═╝     ╚═╝╚══════╝
        \\
        \\
        \\
    ;
    var writer = std.Io.File.stdout().writerStreaming(io, &.{});
    try writer.interface.writeAll(LOGO);
    try writer.interface.flush();
}
