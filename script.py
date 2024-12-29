import typing
import hashlib

from pyscript import fetch, document, window
from pyscript.ffi import create_proxy, to_js


def download_file(data, name, mime_type):
    buffer = window.Uint8Array.new(len(data))
    for pos, b in enumerate(data):
        buffer[pos] = b
    details = to_js({"type": mime_type})

    # This is JS specific
    file = window.File.new([buffer], name, details)
    tmp = window.URL.createObjectURL(file)
    print(tmp)
    dest = document.createElement("a")
    dest.setAttribute("download", name)
    dest.setAttribute("href", tmp)
    dest.click()


startColorRgb = [244, 67, 54]
endColorRgb = [0, 200, 83]
qualityText = document.getElementById("quality-value-text")
qualityProgressBar = document.getElementById("quality-progress-bar")
qualityDescription = document.getElementById("quality-description")
passwordInput = document.getElementById("password")
downloadButton = document.getElementById("download")


report = None
checker = None

down = create_proxy(
    lambda event: (
        download_file(report.encode(), "report.csv", "text/csv")
        if isinstance(report, str)
        else None
    )
)
downloadButton.addEventListener("click", down)


def work(*args, **kwargs):
    global report
    if checker is None:
        return
    passwd = passwordInput.value
    quality, report = checker.check(passwd, 0.1)
    first_line = (
        f"{hashlib.sha1(passwd.encode()).hexdigest()},,,{len(passwd)},{quality:.4f}\n"
    )
    report = first_line + typing.cast(str, to_csv(report))
    qualityText.innerText = f"{quality:.4f} bits"
    progressValue = max(min(quality / 128, 1), 0)
    qualityProgressBar.style.width = f"{progressValue * 100}%"
    qualityDescription.innerText = get_description(quality)
    qualityProgressBar.style.backgroundColor = f"rgb({startColorRgb[0] - (startColorRgb[0] - endColorRgb[0]) * progressValue}, {startColorRgb[1] - (startColorRgb[1] - endColorRgb[1]) * progressValue}, {startColorRgb[2] - (startColorRgb[2] - endColorRgb[2]) * progressValue})"


calc = create_proxy(work)
passwordInput.addEventListener("keyup", calc)
passwordInput.addEventListener("change", calc)


async def read_file(url):
    response = await fetch(url)
    if response.status == 200:
        return await response.text()
    else:
        print(f"Failed to download file: {response.status}")


async def load_checker():
    global checker
    data_adj = Check_Adjacency.load(await read_file("./near.txt"))
    data_pop = Check_Adjacency.load(await read_file("./popular.txt"))
    data_pin = Check_Adjacency.load(await read_file("./pinyin.txt"))
    checker = Checker(
        adj_data=data_adj,
        pinyin_data=data_pin,
        popular_data=data_pop,
    )
    calc()


await load_checker()  # test
