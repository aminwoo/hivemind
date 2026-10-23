# Mate-race rules

How Hivemind scores positions where both teams threaten mate:

- If the opponents already have an unblockable mate in one, Hivemind loses the
  race, even when it has an unblockable mate in one of its own.
- Hivemind's unblockable mate beats the opponents' pending blockable mate in one.
- A blockable mate is terminal only when the defending partner cannot capture
  a piece that would allow a legal blocking drop.
- Races between blockable mates respect side to move and time advantage.
- A stalemated team that is behind on time cannot escape with an irrelevant
  quiet move by the partner: the partner must capture a piece that creates a
  legal drop on the stalemated board. A team ahead on time may still sit.
