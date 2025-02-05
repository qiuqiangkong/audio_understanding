class MIDI2Tokens:
    def __init__(self, fps: float) -> None:
        r"""Convert MIDI events to captions."""
        self.fps = fps

    def __call__(self, data: dict) -> list[str]:
        r"""Convert data of MIDI events to tokens.
        
        Args:
            data: dict

        Outputs:
            tokens: list[str], e.g., ["time_index=15", "name=note_onset", "pitch=36", "velocity=27", ...]
        """

        start_time = data["start_time"]
        duration = data["duration"]
        end_time = start_time + duration

        notes = data["note"]
        pedals = data["pedal"]

        events = []

        for note in notes:

            if note.end < start_time:
                pass

            elif (note.start < start_time) and (start_time <= note.end <= end_time):
                
                events.append([
                    "name=note_offset",
                    "time_index={}".format(round((note.end - start_time) * self.fps)),
                    "pitch={}".format(note.pitch)
                ])

            elif (note.start < start_time) and (end_time < note.end):
                pass

            elif (start_time <= note.start <= end_time) and (start_time <= note.end <= end_time):

                events.append([
                    "name=note_onset",
                    "time_index={}".format(round((note.start - start_time) * self.fps)),
                    "pitch={}".format(note.pitch),
                    "velocity={}".format(note.velocity)
                ])

                events.append([
                    "name=note_offset",
                    "time_index={}".format(round((note.end - start_time) * self.fps)),
                    "pitch={}".format(note.pitch),
                ])

            elif (start_time <= note.start <= end_time) and (end_time < note.end):

                events.append([
                    "name=note_onset",
                    "time_index={}".format(round((note.start - start_time) * self.fps)),
                    "pitch={}".format(note.pitch),
                    "velocity={}".format(note.velocity)
                ])

            elif end_time < note.start:
                pass

            else:
                raise NotImplementedError

        # Sort events by time
        events = self.sort_events(events)

        # Flat tokens
        tokens = self.flat_events(events)

        data.update({"token": tokens})

        return data

    def sort_events(self, events: list[list[str]]) -> list[list[str]]:
        r"""Sort events by time.

        Args:
            events: e.g., [
                ["name=note_offset", "time_index=497", "pitch=69"]
                ["name=note_onset", "time_index=480", "pitch=69", "velocity=62"],
                ...]
            
        Returns:
            sorted_events: e.g., [
                ["name=note_onset", "time_index=480", "pitch=69", "velocity=62"],
                ["name=note_offset", "time_index=497", "pitch=69"]
                ...]
        """
        
        pairs = []

        for event in events:
            pair = self.get_key_value_pair(event)
            pairs.append(pair)

        pairs.sort(key=lambda x: x[0])

        sorted_events = [x[1] for x in pairs]

        return sorted_events

    def get_key_value_pair(self, event: list[str]) -> tuple[str, list[str]]:
        r"""Get key and value pair for sorting events.

        Args:
            event: list[str], e.g., ["name=note_offset", "time_index=56", "pitch=44"]

        Returns:
            key: e.g., "time_index=000056,name=note_offset,pitch=000044"
            value: e.g., ["name=note_offset", "time_index=56", "pitch=44"]
        """

        desired_order = ["time_index", "name", "pitch", "velocity"]
        
        # Sort tokens by desired order
        sorted_tokens = sorted(event, key=lambda x: desired_order.index(x.split('=')[0]))
        # E.g., ["time_index=56", 'name=note_offset', "pitch=44"]

        # Pad 0 for sort
        extended_tokens = [self.extend_token(token) for token in sorted_tokens]
        # E.g., ["time_index=000056", 'name=note_offset', "pitch=000044"]

        key = ",".join(extended_tokens)
        # E.g., "time_index=000056,name=00_note_offset,pitch=000044"

        return key, event

    def extend_token(self, token: str) -> str:
        r"""Left pad values for sorting."""

        key, value = token.split("=")

        if value == "note_offset":
            return "{}=00_{}".format(key, value)

        elif value == "note_onset":
            return "{}=01_{}".format(key, value)

        elif value.isdigit():
            return "{}={:06d}".format(key, int(value))

        else:
            raise NotImplementedError(token)

    def flat_events(self, events: list[list[str]]) -> list[str]:

        tokens = []

        for event in events:
            tokens += event

        return tokens