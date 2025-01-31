import re


class MIDI2TextsSustain:
    def __init__(self, fps: float) -> None:
        r"""Convert MIDI events to captions."""
        self.fps = fps

    def __call__(self, data: dict) -> str:

        start_time = data["start_time"]
        duration = data["duration"]
        end_time = start_time + duration

        notes = data["note"]
        pedals = data["pedal"]

        tokens = []

        for note in notes:

            if note.end < start_time:
                pass

            elif (note.start < start_time) and (start_time <= note.end <= end_time):
                
                tokens.append([
                    "time_index={}".format(round((note.end - start_time) * self.fps)),
                    "name=note_offset",
                    "pitch={}".format(note.pitch)
                ])

            elif (note.start < start_time) and (end_time < note.end):

                tokens.append([
                    "time_index={}".format(round(duration * self.fps)),
                    "name=note_sustain",
                    "pitch={}".format(note.pitch)
                ])

            elif (start_time <= note.start <= end_time) and (start_time <= note.end <= end_time):

                tokens.append([
                    "time_index={}".format(round((note.start - start_time) * self.fps)),
                    "name=note_onset",
                    "pitch={}".format(note.pitch),
                    "velocity={}".format(note.velocity)
                ])

                tokens.append([
                    "time_index={}".format(round((note.end - start_time) * self.fps)),
                    "name=note_offset",
                    "pitch={}".format(note.pitch),
                ])

            elif (start_time <= note.start <= end_time) and (end_time < note.end):

                tokens.append([
                    "time_index={}".format(round((note.start - start_time) * self.fps)),
                    "name=note_onset",
                    "pitch={}".format(note.pitch),
                    "velocity={}".format(note.velocity)
                ])

                tokens.append([
                    "time_index={}".format(round(duration * self.fps)),
                    "name=note_sustain",
                    "pitch={}".format(note.pitch)
                ])

            elif end_time < note.start:
                pass

            else:
                raise NotImplementedError

        # Sort tokens by time
        tokens = self.sort_tokens(tokens)

        # Flat tokens
        tokens = self.flat_tokens(tokens)

        # Convert tokens to sentence
        captions = " ".join(tokens)

        data.update({"caption": caption})

        return data

    def sort_tokens(self, tokens: list[list[str]]) -> list[list[str]]:
        r"""Sort note by time."""
        
        sorted_list = []

        for sub_tokens in tokens:
            
            filled_sub_tokens =  []  # pitch=60 -> pitch=000060

            for token in sub_tokens:    
                
                for number in re.findall(r'\d+', token):
                    token = token.replace(number, "{:06d}".format(int(number)))

                filled_sub_tokens.append(token)

            sorted_list.append({"key": ",".join(filled_sub_tokens), "value": sub_tokens})

        sorted_list.sort(key=lambda x: x["key"])

        sorted_token_list = [x["value"] for x in sorted_list]

        return sorted_token_list

    def flat_tokens(self, tokens: list[list[str]]) -> list[str]:

        flatten_tokens = []

        for sub_tokens in tokens:
            flatten_tokens += sub_tokens

        return flatten_tokens


class MIDI2TextsSustain:
    def __init__(self, fps: float) -> None:
        r"""Convert MIDI events to captions."""
        self.fps = fps

    def __call__(self, data: dict) -> str:

        start_time = data["start_time"]
        duration = data["duration"]
        end_time = start_time + duration

        notes = data["note"]
        pedals = data["pedal"]

        tokens = []

        for note in notes:

            if note.end < start_time:
                pass

            elif (note.start < start_time) and (start_time <= note.end <= end_time):
                
                tokens.append([
                    "time_index={}".format(0),
                    "name=note_sustain",
                    "pitch={}".format(note.pitch)
                ])

                tokens.append([
                    "time_index={}".format(round((note.end - start_time) * self.fps)),
                    "name=note_offset",
                    "pitch={}".format(note.pitch)
                ])

            elif (note.start < start_time) and (end_time < note.end):

                tokens.append([
                    "time_index={}".format(0),
                    "name=note_sustain",
                    "pitch={}".format(note.pitch)
                ])

                tokens.append([
                    "time_index={}".format(round(duration * self.fps)),
                    "name=note_sustain",
                    "pitch={}".format(note.pitch)
                ])

            elif (start_time <= note.start <= end_time) and (start_time <= note.end <= end_time):

                tokens.append([
                    "time_index={}".format(round((note.start - start_time) * self.fps)),
                    "name=note_onset",
                    "pitch={}".format(note.pitch),
                    "velocity={}".format(note.velocity)
                ])

                tokens.append([
                    "time_index={}".format(round((note.end - start_time) * self.fps)),
                    "name=note_offset",
                    "pitch={}".format(note.pitch),
                ])

            elif (start_time <= note.start <= end_time) and (end_time < note.end):

                tokens.append([
                    "time_index={}".format(round((note.start - start_time) * self.fps)),
                    "name=note_onset",
                    "pitch={}".format(note.pitch),
                    "velocity={}".format(note.velocity)
                ])

                tokens.append([
                    "time_index={}".format(round(duration * self.fps)),
                    "name=note_sustain",
                    "pitch={}".format(note.pitch)
                ])

            elif end_time < note.start:
                pass

            else:
                raise NotImplementedError

        # Sort tokens by time
        tokens = self.sort_tokens(tokens)

        # Flat tokens
        tokens = self.flat_tokens(tokens)

        # Convert tokens to sentence
        captions = " ".join(tokens)

        data.update({"caption": caption})

        return data

    def sort_tokens(self, tokens: list[list[str]]) -> list[list[str]]:
        r"""Sort note by time."""
        
        sorted_list = []

        for sub_tokens in tokens:
            
            filled_sub_tokens =  []  # pitch=60 -> pitch=000060

            for token in sub_tokens:    
                
                for number in re.findall(r'\d+', token):
                    token = token.replace(number, "{:06d}".format(int(number)))

                filled_sub_tokens.append(token)

            sorted_list.append({"key": ",".join(filled_sub_tokens), "value": sub_tokens})

        sorted_list.sort(key=lambda x: x["key"])

        sorted_token_list = [x["value"] for x in sorted_list]

        return sorted_token_list

    def flat_tokens(self, tokens: list[list[str]]) -> list[str]:

        flatten_tokens = []

        for sub_tokens in tokens:
            flatten_tokens += sub_tokens

        return flatten_tokens


class MIDI2TextsOnsetOnly:
    def __init__(self, fps: float) -> None:
        r"""Convert MIDI events to captions."""
        self.fps = fps

    def __call__(self, data: dict) -> str:

        start_time = data["start_time"]
        duration = data["duration"]
        end_time = start_time + duration

        notes = data["note"]
        pedals = data["pedal"]

        tokens = []

        for note in notes:

            if note.end < start_time:
                pass

            elif (note.start < start_time) and (start_time <= note.end <= end_time):
                pass

            elif (note.start < start_time) and (end_time < note.end):
                pass

            elif (start_time <= note.start <= end_time) and (start_time <= note.end <= end_time):

                tokens.append([
                    "time_index={}".format(round((note.start - start_time) * self.fps)),
                    "name=note_onset",
                    "pitch={}".format(note.pitch),
                    "velocity={}".format(note.velocity)
                ])

            elif (start_time <= note.start <= end_time) and (end_time < note.end):

                tokens.append([
                    "time_index={}".format(round((note.start - start_time) * self.fps)),
                    "name=note_onset",
                    "pitch={}".format(note.pitch),
                    "velocity={}".format(note.velocity)
                ])

            elif end_time < note.start:
                pass

            else:
                raise NotImplementedError

        # Sort tokens by time
        tokens = self.sort_tokens(tokens)

        # Flat tokens
        tokens = self.flat_tokens(tokens)

        # Convert tokens to sentence
        captions = " ".join(tokens)

        data.update({"caption": captions})

        return data

    def sort_tokens(self, tokens: list[list[str]]) -> list[list[str]]:
        r"""Sort note by time."""
        
        sorted_list = []

        for sub_tokens in tokens:
            
            filled_sub_tokens =  []  # pitch=60 -> pitch=000060

            for token in sub_tokens:    
                
                for number in re.findall(r'\d+', token):
                    token = token.replace(number, "{:06d}".format(int(number)))

                filled_sub_tokens.append(token)

            sorted_list.append({"key": ",".join(filled_sub_tokens), "value": sub_tokens})

        sorted_list.sort(key=lambda x: x["key"])

        sorted_token_list = [x["value"] for x in sorted_list]

        return sorted_token_list

    def flat_tokens(self, tokens: list[list[str]]) -> list[str]:

        flatten_tokens = []

        for sub_tokens in tokens:
            flatten_tokens += sub_tokens

        return flatten_tokens