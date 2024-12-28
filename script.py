from pyscript import fetch
from pyscript import document
from pyscript.ffi import create_proxy


startColorRgb = [244, 67, 54]
endColorRgb = [0, 200, 83]
qualityText = document.getElementById("quality-value-text")
qualityProgressBar = document.getElementById("quality-progress-bar")
qualityDescription = document.getElementById("quality-description")
passwordInput = document.getElementById("password")


checker = None


def work(*args, **kwargs):
    if checker is None:
        return
    quality = checker.check(passwordInput.value, 0.1)[0]
    qualityText.innerText = f"{quality:.4f} bits"
    progressValue = max(min(quality / 128, 1), 0)
    qualityProgressBar.style.width = f"{progressValue * 100}%"
    qualityDescription.innerText = get_description(quality)
    qualityProgressBar.style.backgroundColor = f"rgb({startColorRgb[0] - (startColorRgb[0] - endColorRgb[0]) * progressValue}, {startColorRgb[1] - (startColorRgb[1] - endColorRgb[1]) * progressValue}, {startColorRgb[2] - (startColorRgb[2] - endColorRgb[2]) * progressValue})"


proxy = create_proxy(work)
passwordInput.addEventListener("keyup", proxy)
passwordInput.addEventListener("change", proxy)


async def download_file(url):
    response = await fetch(url)
    if response.status == 200:
        return await response.text()
    else:
        print(f"Failed to download file: {response.status}")


async def load_checker():
    global checker
    data_adj = Check_Adjacency.load(await download_file("./near.txt"))
    data_pop = Check_Adjacency.load(await download_file("./popular.txt"))
    data_pin = Check_Adjacency.load(await download_file("./pinyin.txt"))
    checker = Checker(
        adj_data=data_adj,
        pinyin_data=data_pin,
        popular_data=data_pop,
    )
    proxy()


await load_checker()  # test
