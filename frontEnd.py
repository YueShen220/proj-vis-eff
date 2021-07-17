#!/usr/bin/python
import webbrowser
import cgi
import os
import cgitb
cgitb.enable()
#import API.py


# Different files for each group. 1 data preparation
# Importing other files, make classes. OOP. Calling file in python.
# Functions

animations = {
    "left": """
                left: 200px;
            }
    
            @keyframes actor {
                0% {left:200px; transform: rotate(0deg);}
                30% {left:450px; transform: rotate(-20deg);}
                50% {left:700px; transform: rotate(40deg);}
                80% {left:100px; transform: rotate(-20deg);}
                100% {left:200px; transform: rotate(0deg);}
            }
            """,
    "right": """
            right: 200px;
            }
    
            @keyframes actor {
                0% {right:200px; transform: rotate(0deg);}
                30% {right:450px; transform: rotate(-20deg);}
                50% {right:700px; transform: rotate(40deg);}
                80% {right:100px; transform: rotate(-20deg);}
                100% {right:200px; transform: rotate(0deg);}
            }
            """
}

animation = animations["right"]


def display(List: animation):
    html = """
    <!DOCTYPE html>
    <html lang="en">

    <!--

        colors:
            #790909
            #000000
            #FBFFFE
            #5C6F68
            #F49D37
    -->

    <head>
        <style>
            @font-face {
                font-family: 'ZILAPGEOMETRIK';
                src: url('../proj-vis-eff/fonts/ZILAPGEOMETRIK.svg') format('svg'),
                    url('../proj-vis-eff/fonts/ZILAPGEOMETRIK.ttf') format('truetype'),
                    url('../proj-vis-eff/fonts/ZILAPGEOMETRIK.woff') format('woff');
                font-weight: normal;
                font-style: normal;
            }

            @font-face {
                font-family: 'ZilapGeometrik-VnKy';
                src: url('../proj-vis-eff/fonts/ZilapGeometrik-VnKy.eot');
                src: url('../proj-vis-eff/fonts/ZilapGeometrik-VnKy.eot?#iefix') format('embedded-opentype');
                font-weight: normal;
                font-style: normal;
            }



            body {
                font-family: 'ZILAPGEOMETRIK';
                margin: 0;
                padding: 0;
                background-color: #5C6F68;
                color: #FBFFFE;
            }

            h1,
            h2 {
                margin: 20px auto 0 70px;
            }

            h1{
                font-size: 75px;
                text-decoration: underline;
            }

            h2{
                font-size: 60px;
            }


            main {
                box-sizing: border-box;
                width: calc(100%-100px);
                color: #5C6F68;
                display: grid;
                grid-template-rows: 600px 800px;
            }

            form {
                text-align: center;
                box-sizing: border-box;
                height: 450px;
                padding: 200px min(250px, 20%);
                margin: 100px 70px;
                background-color: #F49D37;
                outline: 2px dashed black;
                outline-offset: -10px;
                border-radius: 25px;
            }

            form label {
                font-size: 50px;
                color: #FBFFFE;
            }

            .box__dragndrop,
            .box__uploading,
            .box__success,
            .box__button,
            .box__error {
                display: none;
            }

            .box__file {
                width: 0.1px;
                height: 0.1px;
                opacity: 0;
                overflow: hidden;
                position: absolute;
                z-index: -1;
            }







            .theater{
                background-color: #000000;
                margin-top: 50px;
                height: 750px;
                width: 100%;
                outline: 2px dashed black;
            }

            .alive{
                position: absolute;
                background-color: white;
                padding: 50px;
                width: 100px;
                height: 100px;
                box-sizing: border-box;
                border-radius: 25px;
                animation: actor 5s linear 2s infinite;
                top: 1200px;
    """

    # Set animation:
    html += animation

    html += """
        </style>
    </head>

    <body>
        <h1> AI Launch Lab </h1>
        <h2> OAYE VFX: making visuals from gestures</h2>
        <main>
            <form class="box" method="post" action="" enctype="multipart/form-data">
                <div class="box__input">
                    <input class="box__file" type="file" name="files[]" id="file"
                        data-multiple-caption="{count} files selected" multiple />
                    <label for="file"><strong>Choose a JPG file</strong><span class="box__dragndrop"> or drag it
                            here</span>.</label>
                    <button class="box__button" type="submit">Upload</button>
                </div>
                <div class="box__uploading">Uploading…</div>
                <div class="box__success">Done!</div>
                <div class="box__error">Error! <span></span>.</div>
            </form>
            <div class="theater">
                <div class="alive">

                </div>
            </div>
        </main>
    </body>

    </html>"""

    with open('helloworld.html', 'w') as f:
        message = html

        f.write(message)

    webbrowser.open_new_tab('helloworld.html')

display(animation)




# def retrieveVideo(File: video):


form = cgi.FieldStorage()


# Get filename here.
#fileitem = form['filename']

# Test if the file was uploaded
# if fileitem.filename:
# strip leading path from file name to avoid
# directory traversal attacks
#   fn = os.path.basename(fileitem.filename)
#  open('/tmp/' + fn, 'wb').write(fileitem.file.read())
# message = 'The file "' + fn + '" was uploaded successfully'
# else:
#   message = 'No file was uploaded'

# print("""\
# Content-Type: text/html\n
# <html>
# <body>
#   <p>%s</p>
# </body>
# </html>
# """ % (message,))
