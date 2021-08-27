import pyaudio


def prompt_device_index(p: pyaudio.PyAudio) -> int:
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')

    if not num_devices:
        # no input devices detected
        print("No input devices detected!")
        print("Terminating program")
        raise IOError

    while True:

        print("Available input devices:")

        valid_ids = []

        for i in range(0, num_devices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                valid_ids.append(i)
                print(f"Input Device {i} - {p.get_device_info_by_host_api_device_index(0, i).get('name')}")

        input_raw = input("Please select a device by id: ")

        if not input_raw:
            # default to using the first device
            return 0
        else:
            # attempt to convert the input to a integer
            try:
                selected_id = int(input_raw)
                # check if the inputted id is valid
                if selected_id not in valid_ids:
                    print(f"{selected_id} is not a valid id!")
                else:
                    return selected_id
            except ValueError:
                print(f"\"{input_raw}\" is not a valid integer!")

            print("Please try again.")


def byte_width_to_format_char(byte_width: int, signed: bool = True) -> str:
    format_char_map = {
        1: "b",
        2: "h",
        4: "i",
        8: "q"
    }

    if byte_width not in format_char_map:
        raise ValueError(f"Unsupported byte width {byte_width}")

    format_char = format_char_map[byte_width]

    if not signed:
        format_char = format_char.upper()

    return format_char
