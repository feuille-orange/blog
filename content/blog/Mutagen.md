+++

title = "Mutagen：好用的端口转发与远程文件同步工具"

date = "2025-08-19"

[taxonomies]

tags = ["IT Operations", "Tools"]

+++

`Mutagen` 是好用的命令行 `SSH` 转发与 `SCP` 目录同步软件，见 [mutagen-io/mutagen: Fast file synchronization and network forwarding for remote development](https://github.com/mutagen-io/mutagen)，主要支持以下功能：

- 文件同步：本地与远程、远程与远程（通过本地中转）、本地与 Docker 等
- 长期端口转发：只要创建后一直保持端口转发

## 安装

从 [Releases · mutagen-io/mutagen](https://github.com/mutagen-io/mutagen/releases) 中下载二进制文件并放到 `PATH` 中。

## 快速上手：端口转发

端口转发的功能由 `mutagen forward` 控制，具体而言：

- `mutagen forward create --name=web-app tcp:localhost:8080 docker://devcontainer:tcp:localhost:1313` 将 Docker 的 `1313` 端口映射到本地的 `8080` （支持 SSH 配置）
- `mutagen forward list` ：列出端口转发列表
- `mutagen forward monitor <name>` ：监控端口转发
- `mutagen forward terminate <name>` ：取消端口转发
- `mutagen forward pause/resume <name>` ：暂停与恢复端口转发

``` bash
$ mutagen forward --help
Create and manage network forwarding sessions

Usage:
  mutagen forward [flags]
  mutagen forward [command]

Available Commands:
  create      Create and start a new forwarding session
  list        List existing forwarding sessions and their statuses
  monitor     Display streaming session status information
  pause       Pause a forwarding session
  resume      Resume a paused or disconnected forwarding session
  terminate   Permanently terminate a forwarding session

Flags:
  -h, --help   Show help information

Use "mutagen forward [command] --help" for more information about a command.
```

## 快速上手：文件同步

文件同步相关功能由 `mutagen sync` 命令控制，具体包括：

- `mutagen sync create --name=web-app-code ~/project user@example.org:~/project` ：创建同步（支持 SSH 配置）
- `mutagen sync list` ：列出同步会话
- `mutagen sync monitor <name>` ：监控文件同步
- `mutagen sync terminate <name>` ：取消同步
- `mutagen sync pause/resume <name>` ：暂停与恢复同步

``` bash
$ mutagen.exe sync --help
Create and manage file synchronization sessions

Usage:
  mutagen sync [flags]
  mutagen sync [command]

Available Commands:
  create      Create and start a new synchronization session
  list        List existing synchronization sessions and their statuses
  monitor     Display streaming session status information
  flush       Force a synchronization cycle
  pause       Pause a synchronization session
  resume      Resume a paused or disconnected synchronization session
  reset       Reset synchronization session history
  terminate   Permanently terminate a synchronization session

Flags:
  -h, --help   Show help information

Use "mutagen sync [command] --help" for more information about a command.
```

