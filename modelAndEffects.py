# Authors:
#             GOMEZ MAQUEO ANAYA, Santiago
#             HEBRARD, Emilie
#             SHEN, Yue


# Date of start:            19/06/2021
# Date of finish:           -/-/-




# This is the main file for the project.
# The idea is to have a Machine Learning model predict labels 
# from gestures, and then make the correspondance between that 
# (as well as certainmvariables from our datapoints, like acceleration)
# and feelings.

# We'll then take these feelings (strings) and their intensities (floats)
# and generate/select corresponding visual effects through html/css.



# -- IMPORTS AND LIBRARIES --







# -- MODEL SECTION OF THE CODE --
# The model we train
model = {}

# The database we use for training
database = [1,2,3,4,5]

















# -- DISPLAY SECTION OF THE CODE --

# This determines how our site looks
cssText = """
        html, 
        body {
            height: 100%;
            margin: 0;
            overflow: hidden;
        }

        body{
            background-color: #212529;
        }

        p{
            font-size:20px;
            color: white;
        }
"""


# This is the structure of our site/display
htmlText=""

htmlHeadStart = """ 
<!DOCTYPE html> 
<html lang="en"> 

<head>
    <title>Welcome</title>
    <link rel="shortcut icon" type="image/png" href="">
    <meta charset="UTF-8"> 
    <meta name="viewport" content="width=device-width, initial-scale=1"> 
    <style>"""

htmlHeadStart += cssText

htmlHeadEnd="""
    </style>
</head>

"""

htmlBodyStart="""
<body>

"""

htmlBodyEnd="""
</body>\n

</html>"""


htmlText= htmlHeadStart + htmlHeadEnd + htmlBodyStart + htmlBodyEnd

# And now we display the html file... somehow. TODO