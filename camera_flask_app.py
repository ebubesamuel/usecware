import functools
from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
import face_recognition
from threading import Thread

import sqlite3
#import click


from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import session
from flask import url_for
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash

#rec_frame
global capture, grey,rec_frame, switch, neg, face, rec, out
capture=0
grey=0
neg=0
face=1
id = 1 
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('static/shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')
app.config.from_mapping(
        # a default secret that should be overridden by instance config
        SECRET_KEY="dev",
        # store the database in the instance folder
        DATABASE=os.path.join(app.instance_path, "db.sqlite"),
    )

# if test_config is None:
#     # load the instance config, if it exists, when not testing
app.config.from_pyfile("config.py", silent=True)
# else:
#     # load the test config if passed in
#     app.config.update(test_config)

# ensure the instance folder exists
try:
    os.makedirs(app.instance_path)
except OSError:
    pass


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    # try:
    #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #         #roegion_of_interest_gray = gray[y:y+w, x:x+w]
    #         roegion_of_interest_color = frame[y:y+h, x:x+w]
    #         #cv2.imwrite('shots/captured_image.jpg', roegion_of_interest_color)
    # except Exception as e:
    #     print(e)
        
    # return frame
 
 
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
            return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 5)
        # frame=frame[startY:endY, startX:endX]
        # (h, w) = frame.shape[:2]
        # r = 480 / float(h)
        # dim = ( int(w * r), 480)
        # frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame

 

def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read() 
        #print(success)
        if success:
            if(face):                
                frame= detect_face(frame)
            if(id):
                x, y, width, height = 100, 100, 800, 500

                # Draw the rectangle on the frame
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)            
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(e)
                
        else:
            pass


def login_required(view):
    """View decorator that redirects anonymous users to the login page."""

    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for("login"))

        return view(**kwargs)

    return wrapped_view


@app.route('/')
#@login_required
def index():
    global id, face
    id = 1
    #face = 0
    return render_template('camera/index.html')

@app.route('/records')
@login_required
def records():
    db = get_db()
    records= db.execute(
        "SELECT id, image1, image2, result, created_at"
        " FROM record"
    ).fetchall()
    return render_template('camera/records.html', records = records)
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera, id, face
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            _ , frame = camera.read()
            frame = detect_face(frame)
            x, y, width, height = 100, 100, 800, 500

            # Draw the rectangle on the frame
            #cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            roi = frame[y:y+height, x:x+width]
            now = datetime.datetime.now()
            p = os.path.sep.join(['static/shots', "shot_{}.png".format(str(now).replace(":",''))])
            cv2.imwrite(p, roi)
            print(p)
            id = 0
            #face = 1
            return render_template('camera/index2.html', img1 = p)
            
            
            # global capture
            # capture=1
        
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        id = 1
        #face = 0
        return render_template('camera/index.html')
    return render_template('camera/index.html')



@app.route('/requests2',methods=['POST','GET'])
def tasks2():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            _ , frame = camera.read()
            frame = detect_face(frame)
            now = datetime.datetime.now()
            p = os.path.sep.join(['static/shots', "shot_{}.png".format(str(now).replace(":",''))])
            cv2.imwrite(p, frame)
            result = IsFaceMatching(request.args.get('img1'), p)
            db = get_db()
            db.execute("INSERT INTO record (image1, image2, result) VALUES (?, ?, ?)",
                (request.args.get('img1'), p, result))
            db.commit()
            
            return render_template('camera/index3.html', result = result)
            
            
            # global capture
            # capture=1
        
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        
                                       
    elif request.method=='GET':
        return render_template('camera/index2.html')
    return render_template('camera/index2.html')


def IsFaceMatching(img1, img2):
    print(img1)
    print(img2)
    img_enyi = face_recognition.load_image_file(img1)
    img_enyi = cv2.cvtColor(img_enyi, cv2.COLOR_BGR2RGB)

    #---------------- Detecting Face----------------
    face = face_recognition.face_locations(img_enyi) [0]
    #copy = img_enyi_rgb.copy()

    #-----------converting image into encodings
    train_encode = face_recognition.face_encodings(img_enyi) [0]

    #---------------Drawing boxes around face-------------------
    #new_height = face[2] + 20  # Increase the height by 20 pixels (you can adjust this value)
    test = face_recognition.load_image_file(img2)
    test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    face2 = face_recognition.face_locations(test) [0]
    test_encode = face_recognition.face_encodings(test)[0]
    match = face_recognition.compare_faces([train_encode], test_encode)
    if match[0]:  # Check if the first element in the list is True
        return True  # Log something to the console if it's a match
    else:
        return False  # Log something to the console if it's not a match
    # cv2.rectangle(img_enyi, (face[3], face[0]), (face[1], face[2]), (255,0,255), 5)
    # cv2.imshow('img_enyi', img_enyi)
    # cv2.rectangle(test, (face2[3], face2[0]), (face2[1], face2[2]), (255,0,255), 5)
    # cv2.imshow('test', test)


    #cv2.imshow('Enyi', img_enyi_rgb)
    #cv2.waitKey(0)

    #img_enyi = face_recognition



#SEPERATED CODE
#all routes had bp




# def login_required(view):
#     """View decorator that redirects anonymous users to the login page."""

#     @functools.wraps(view)
#     def wrapped_view(**kwargs):
#         if g.user is None:
#             return redirect(url_for("auth.login"))

#         return view(**kwargs)

#     return wrapped_view

#before_app_request
@app.before_request
def load_logged_in_user():
    """If a user id is stored in the session, load the user object from
    the database into ``g.user``."""
    user_id = session.get("user_id")

    if user_id is None:
        g.user = None
    else:
        g.user = (
            get_db().execute("SELECT * FROM user WHERE id = ?", (user_id,)).fetchone()
        )


@app.route("/register", methods=("GET", "POST"))
def register():
    """Register a new user.

    Validates that the username is not already taken. Hashes the
    password for security.
    """
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        db = get_db()
        error = None

        if not username:
            error = "Username is required."
        elif not password:
            error = "Password is required."

        if error is None:
            try:
                db.execute(
                    "INSERT INTO user (username, password) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                db.commit()
            except db.IntegrityError:
                # The username was already taken, which caused the
                # commit to fail. Show a validation error.
                error = f"User {username} is already registered."
            else:
                # Success, go to the login page.
                return redirect(url_for("login"))

        flash(error)

    return render_template("auth/register.html")


@app.route("/login", methods=("GET", "POST"))
def login():
    """Log in a registered user by adding the user id to the session."""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        db = get_db()
        error = None
        user = db.execute(
            "SELECT * FROM user WHERE username = ?", (username,)
        ).fetchone()

        if user is None:
            error = "Incorrect username."
        elif not check_password_hash(user["password"], password):
            error = "Incorrect password."

        if error is None:
            # store the user id in a new session and return to the index
            session.clear()
            session["user_id"] = user["id"]
            return redirect(url_for("index"))

        flash(error)

    return render_template("auth/login.html")


@app.route("/logout")
def logout():
    """Clear the current session, including the stored user id."""
    session.clear()
    return redirect(url_for("index"))



#SEPERATED CODE


def get_db():
    """Connect to the application's configured database. The connection
    is unique for each request and will be reused if this is called
    again.
    """
    with app.app_context():
        if "db" not in g:
            g.db = sqlite3.connect(
                app.config["DATABASE"], detect_types=sqlite3.PARSE_DECLTYPES
            )
            g.db.row_factory = sqlite3.Row

        return g.db


def close_db(e=None):
    """If this request connected to the database, close the
    connection.
    """
    db = g.pop("db", None)

    if db is not None:
        db.close()


def init_db():
    """Clear existing data and create new tables."""
    db = get_db()

    with app.open_resource("schema.sql") as f:
        db.executescript(f.read().decode("utf8"))


# @click.command("init-db")
# def init_db_command():
#     """Clear existing data and create new tables."""
#     init_db()
#     click.echo("Initialized the database.")


def init_app(app):
    """Register database functions with the Flask app. This is called by
    the application factory.
    """
    app.teardown_appcontext(close_db)
    #app.cli.add_command(init_db_command)


if __name__ == '__main__':
    if not os.path.exists(os.path.join(app.instance_path, "db.sqlite")):
        init_db()
    app.run(debug=True)
    
    
#camera.release()
#cv2.destroyAllWindows()     