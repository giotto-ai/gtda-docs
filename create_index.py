
import os

s = "\n".join(["<!DOCTYPE html>", "<html>", "<body>"])

s += "\n<form action='???'>"
s += "\n<label>Choose a version: </label>"
s += "\n <select name='versions'>"
available_folders = filter(lambda x: os.path.isdir(x) and not(x.startswith('.')),
                           os.listdir('.'))
s += "\n".join([f"<option value='{dir_n}/index.html'>{dir_n}</option>" for dir_n in available_folders])
s += "\n </select>"
s += "\n<input type='submit' value='Submit'>"
s += "\n</form>"
s += "\n".join(["</html>", "</body>"])

with open('index_2.html', 'w') as f:
    f.write(s)
