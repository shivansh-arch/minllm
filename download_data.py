import urllib.request

urllib.request.urlretrieve(
    "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "holmes.txt"
)

print("Downloaded holmes.txt")