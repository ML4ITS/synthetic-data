import os
import shutil
import subprocess

import pandas as pd


def df_to_csv(df: pd.DataFrame) -> None:
    return df.to_csv(index=False).encode("utf-8")


def prettify_name(name: str) -> str:
    return name.replace("_", " ").title()


def strtobool(val: str) -> bool:
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"Invalid truth value '{val}'")


def name_to_alias(name: str) -> str:
    return name.lower().replace(" ", "_")


def make_tmpdir(tmpdir: str) -> None:
    os.makedirs(tmpdir, exist_ok=True)


def del_tmpdir(tmpdir: str) -> None:
    shutil.rmtree(tmpdir)


def create_gif_from_image_folder(
    folder: str, filename: str, fps: int, loop: int = 0
) -> None:
    """Create a gif from a folder of images.

    NOTE: %04d should match the total number of expected image range
    Example:
          %03d will zero padded to 3 digits, so (0-999)
          %04d will zero padded to 4 digits, so (0-9999)
          %05d will zero padded to 5 digits, so (0-99999)



    Args:
        folder (str): Folder containing images.
        filename (str): Name of the gif file.
        fps (int): Frames per second.
        loop (int, optional): loop count. Defaults to 0.
            if loop is 0, the gif will loop forever.
            if loop is 1, the gif will loop once.
            if loop is 2, the gif will loop twice.

    """

    assert ".gif" in filename, f"Filename '{filename}' must end with .gif"
    assert os.path.isdir(folder), f"Folder '{folder}' does not exist"
    assert 0 <= fps <= 60, "FPS must be between 0 and 60"
    assert 0 <= loop <= 2, "Loop must be between 0 and 2"

    cmd = f"ffmpeg -f image2 -framerate {fps} -i {folder}/frame%04d.jpg -loop -{loop} -pix_fmt bgr8 {filename}"
    subprocess.run(
        cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
