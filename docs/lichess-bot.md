# Lichess BOT

The `lichess-bot` command accepts Chess (`standard`), Crazyhouse, Antichess,
Chess960, Atomic, and Three-check challenges and plays all of them with
`engine/models/twin-s-noattn.onnx`, with draw contempt set to zero. It accepts one incoming
game at a time, declines other variants and UltraBullet, reconnects interrupted
streams, and reserves clock time for network latency. Bughouse remains
available through the engine's UCI interface; Lichess does not offer Bughouse.

Build the engine first using [the build guide](../engine/README.md). For CPU:

```sh
uv run hivemind lichess-bot --token-file /path/to/private/lichess.token --check
uv run hivemind lichess-bot --token-file /path/to/private/lichess.token
```

The default executable is `engine/build-ort/hivemind.bin`. To use NVIDIA:

```sh
uv run hivemind lichess-bot --engine engine/build-ninja/hivemind \
    --model engine/models/twin-s-noattn.onnx \
    --token-file /path/to/private/lichess.token
```

The token needs `bot:play`; `challenge:write` is useful for creating test games.
Use an existing BOT account. The command checks the account title and never
upgrades accounts automatically. Lichess's account upgrade is irreversible;
see [the official Bot API](https://lichess.org/api#tag/bot).

Keep the token outside the repository in a file readable only by its owner
(`chmod 600`), or provide it through `LICHESS_TOKEN`. It is never added to logs
or passed as a command-line argument. `--move-time-ms` sets the maximum time
per move (500 ms by default); `--run-seconds` bounds a listening test.

For continuous operation on this Linux workspace, the template
[`tools/hivemind-lichess-bot.service`](../tools/hivemind-lichess-bot.service)
uses the TensorRT executable and `%h/.config/hivemind/lichess.token`. It assumes
the checkout is at `%h/hivemind`; adjust the paths or choose the CPU executable
for other installations. Install it under `~/.config/systemd/user/`, then:

```sh
systemctl --user daemon-reload
systemctl --user enable --now hivemind-lichess-bot
journalctl --user -u hivemind-lichess-bot -f
systemctl --user stop hivemind-lichess-bot
```

Unit tests exercise challenge acceptance and move-history reconstruction.
Live testing should use casual games. The API integration follows Lichess's
[incoming event stream](https://lichess.org/api#tag/board/GET/api/stream/event),
[challenge acceptance](https://lichess.org/api#tag/challenges/POST/api/challenge/{challengeId}/accept),
and [Bot game stream](https://lichess.org/api#tag/bot/GET/api/bot/game/stream/{gameId}).
