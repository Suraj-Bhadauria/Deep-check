from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from fpdf import FPDF
from moviepy import VideoFileClip
import torch
import os
import uuid
import cv2
from PIL import Image
import exifread
import hashlib
from PIL.ExifTags import TAGS
from datetime import datetime
from moviepy.video.io.VideoFileClip import VideoFileClip
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from grad_cam import generate_heatmap, overlay_heatmap



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

db = SQLAlchemy(app)

class ImageModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.String(100))
    image_metadata=db.Column(db.Text, nullable=True)#YE JATIN KA CODE HAI
    
class VideoModel(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    filename = db.Column(db.String(500), nullable = False)
    prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.String(100))
    image_metadata=db.Column(db.Text, nullable=True)

class InfoMedel(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(500), nullable=False)
    designation = db.Column(db.String(500), nullable=False)
    caseID = db.Column(db.String(500), nullable=False)
    org = db.Column(db.String(500), nullable=False)
    phone = db.Column(db.String(500), nullable=False)
    snum = db.Column(db.String(500), nullable=False)

admin_username = "admin"
admin_password = "123"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mkv', 'avi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#parent model (Mesonet) declaration from class Meso4() to return x shall be same on further commits
class Meso4(nn.Module):
    def __init__(self):
        super(Meso4, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(8, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(16, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 2)
        self.softmax = nn.Softmax(dim=1)

        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # return x
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        self.activations = x

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.softmax(x)
    
    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 'model' is name for image classification model of the path "Meso2V1_1.13.pth"
model = Meso4()
state_dict = torch.load("Meso2V1_1.13.pth", map_location=device)
new_state_dict = {key.replace("meso.", "").replace("fc_video.", ""): value for key, value in state_dict.items()}
model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()

# 'model0' is name for video classification model of the path "Meso2V1_1.15(vid).pth"
model0 = Meso4()
state_dict_video = torch.load("Meso2V1_1.15(vid).pth", map_location=device)
new_state_dict_video = {key.replace("meso.", "").replace("fc_video.", ""): value for key, value in state_dict_video.items()}
model0.load_state_dict(new_state_dict_video, strict=False)
model0.to(device)
model0.eval()

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

def apply_grad_cam(model, image_tensor):
    # Forward pass
    output = model(image_tensor)
    predicted_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, predicted_class].backward()

    # Get gradients and activations
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = model.get_activations().detach()

    # Weight the activations by the gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Generate heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu().numpy(), 0)
    heatmap /= np.max(heatmap)

    return heatmap

def frame_extraction(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval == 0:
            frames.append(frame)

    cap.release()
    return frames

def report_pdf(filename, prediction,metadata):
    pdf = FPDF()
    pdf.add_page()

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(200, 10, "REPORT ON IMAGE FILE", ln=True, align="C")

    pdf.line(10, 20, 200, 20)

    pdf.image(image_path, x=10, y=30, w=70, h=40)

    pdf.ln(60)
    pdf.line(10, 75, 200, 75)

    pdf.line(10, 90, 200, 90)


    #1.general info block
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"General Information", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"Report ID: ",ln=True)
    pdf.cell(0, 10, f"Date and Time: {metadata.get('Date Created', 'N/A')}",ln=True)
    pdf.cell(0, 10, f"Investigation Case ID: ",ln=True)
    pdf.cell(0, 10, f"Analyst Name: ",ln=True)
    pdf.cell(0, 10, f"Organisation Name: ",ln=True)

    pdf.ln(5)

    #2.input file details
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"Input File Details", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"File Name-{filename}",ln=True)
    pdf.cell(0, 10, f"File Type:{metadata.get('File Format', 'N/A')}",ln=True)
    pdf.cell(0, 10, f"File Size:{metadata.get('File Size', 'N/A')}",ln=True)
    pdf.cell(0, 10, f"Resolution: ",ln=True)
    pdf.cell(0, 10, f"Duration(for video): ",ln=True)
    pdf.cell(0, 10, f"Frame Rate(for video): ",ln=True)

    pdf.ln(5)
    #3.Deepfake analysis result
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"Deepfake Analysis Results", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"Confidence Score: ",ln=True)
    pdf.cell(0, 10, f"Deepfake Likelihood: ",ln=True)
    pdf.cell(0, 10, f"Suspicious Regions Identified: ",ln=True)

    pdf.ln(5)

    pdf.add_page()

    #4.Suspicious Frames(for video reports)
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"Suspicious Frames(for video reports)", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"Number of suspicious frames: ",ln=True)
    pdf.cell(0, 10, f"Timestamps of suspicious frames: ",ln=True)

    pdf.ln(5)

    #5.Metadata and forensics details
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"Metadata and Forensic Details", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"File Metadata: ",ln=True)
    pdf.cell(0, 10, f"EXIF Data(for images): ",ln=True)
    pdf.cell(0, 10, f"Hash Value(SHA-256):{metadata.get('Hash Value', 'N/A')} ",ln=True)
    pdf.cell(0, 10, f"Compression Artifacts: ",ln=True)
    pdf.cell(0, 10, f"Frame Manipulation Detection: ",ln=True)

    pdf.ln(5)

    #6.Obeservations 
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"Observations and Expert Remarks", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"Summary of Findings: ",ln=True)
    pdf.cell(0, 10, f"Possible Manipulations Techniques: ",ln=True)
    pdf.cell(0, 10, f"Confidence Level: ",ln=True)
    pdf.cell(0, 10, f"Suggested Next Steps: ",ln=True)

    pdf.ln(5)

    #7.Conclusions and signature
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"Conclusion and Signature", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"Final Verdict: ",ln=True)
    pdf.cell(0, 10, f"Forensic Analyst Signature and Stamp: ",ln=True)


    # pdf.set_font("Arial", size=14)
    # pdf.cell(0, 10, f"Filename - {filename}", ln=True)
    # pdf.cell(0, 10, f"Prediction by Meso2V1 - {prediction}", ln=True)
    # pdf.cell(0, 10, f"File Format - ", ln=True)
    # pdf.cell(0, 10, f"")


    report_path = os.path.join(app.config['UPLOAD_FOLDER'], f"Report on {filename}.pdf")
    pdf.output(report_path)

    return(report_path)

# backend for video pdf button
def report_pdf0(filename, prediction, fake_count, real_count, avg_confidence, metadata):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", style="B", size=20)
    pdf.cell(200, 10, "REPORT ON VIDEO FILE")

    # pdf.line(10, 30, 210, 30)

    # pdf.set_font("Arial", size=14)
    # pdf.cell(0, 10, f"Filename - {filename}", ln=True)
    # pdf.cell(0, 20, f"Prediction - {prediction}", ln=True)
    # pdf.cell(0, 30, f"Fake frame count - {fake_count}")
    # pdf.cell(0, 40, f"Real frame count - {real_count}")
    # pdf.cell(0, 50, f"Confidence - {avg_confidence}")

    # -----video pdf report---------
    
    pdf.line(10, 20, 200, 20)

    

    pdf.ln(30)
    # pdf.line(10, 75, 200, 75)

    # pdf.line(10, 90, 200, 90)


    #1.general info block
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"General Information", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"Report ID: 001",ln=True)
    pdf.cell(0, 10, f"Date and Time:{metadata.get('Date and Time Created', 'N/A')} ",ln=True)
    pdf.cell(0, 10, f"Investigation Case ID: ",ln=True)
    pdf.cell(0, 10, f"Analyst Name: ",ln=True)
    pdf.cell(0, 10, f"Organisation Name: ",ln=True)

    pdf.ln(5)

    #2.input file details
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"Input File Details", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"File Name-{filename} ",ln=True)
    pdf.cell(0, 10, f"File Type:{metadata.get('File Format','N/A')} ",ln=True)
    pdf.cell(0, 10, f"File Size:{metadata.get('File Size','N/A')} ",ln=True)
    pdf.cell(0, 10, f"Resolution: -",ln=True)
    pdf.cell(0, 10, f"Duration(for video):{metadata.get('Video Duration','N/A')} ",ln=True)
    pdf.cell(0, 10, f"Frame Rate(for video): -",ln=True)

    pdf.ln(5)
    #3.Deepfake analysis result
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"Deepfake Analysis Results", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"Confidence Score: {avg_confidence*100:.2f}",ln=True)
    pdf.cell(0, 10, f"Deepfake Likelihood: ",ln=True)
    pdf.cell(0, 10, f"Suspicious Regions Identified: ",ln=True)

    pdf.ln(5)

    pdf.add_page()

    #4.Suspicious Frames(for video reports)
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"Suspicious Frames(for video reports)", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"Number of suspicious frames: {fake_count}",ln=True)
    pdf.cell(0, 10, f"Timestamps of suspicious frames: ",ln=True)

    pdf.ln(5)

    #5.Metadata and forensics details
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"Metadata and Forensic Details", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"File Metadata: ",ln=True)
    pdf.cell(0, 10, f"EXIF Data(for images): ",ln=True)
    pdf.cell(0, 10, f"Hash Value(SHA-256):{metadata.get('Hash Value','N/A')} ",ln=True)
    pdf.cell(0, 10, f"Compression Artifacts: ",ln=True)
    pdf.cell(0, 10, f"Frame Manipulation Detection: ",ln=True)

    pdf.ln(5)

    #6.Obeservations 
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"Observations and Expert Remarks", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"Summary of Findings: ",ln=True)
    pdf.cell(0, 10, f"Possible Manipulations Techniques: ",ln=True)
    pdf.cell(0, 10, f"Confidence Level: ",ln=True)
    pdf.cell(0, 10, f"Suggested Next Steps: ",ln=True)

    pdf.ln(5)

    #7.Conclusions and signature
    pdf.set_font('Arial', style='B', size=20)
    pdf.cell(0, 10, f"Conclusion and Signature", ln=True)

    pdf.set_font('Arial', style='B', size=14)
    pdf.cell(0, 10, f"Final Verdict: ",ln=True)
    pdf.cell(0, 10, f"Forensic Analyst Signature and Stamp: ",ln=True)


    # -------------------------------

    report_path0 = os.path.join(app.config['UPLOAD_FOLDER'], f"Report on {filename}.pdf")
    pdf.output(report_path0)
    print(f"PDF saved at: {report_path0}")

    return(report_path0)



@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

@app.route("/help", methods=["GET"])
def help():
    return render_template("help.html")

@app.route("/video", methods=["GET", "POST"])
def video():
    return render_template("video.html")

@app.route("/history", methods=["GET", "POST"])
def history():
    return render_template("history.html")

@app.route("/choose", methods = ["GET", "POST"])
def choose():
    return render_template("choose.html")

@app.template_global()
def get_users():
    return InfoMedel.query.all()

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username ==  admin_username and password == admin_password:
            return redirect(url_for('info'))
    
    return render_template("login.html")

@app.route("/info", methods=["GET", "POST"])
def info():
    if request.method == 'POST':
        name = request.form["name"]
        designation = request.form["designation"]
        caseID=request.form["caseID"]
        org = request.form["org"]
        phone = request.form["phone"]
        snum = request.form["snum"]

        info = InfoMedel(name=name, designation=designation,caseID=caseID, org = org, phone=phone, snum=snum)
        db.session.add(info)
        db.session.commit()
        return redirect(url_for('choose'))

    return render_template("info.html")

@app.route("/comparator", methods=["GET", "POST"])
def comparator():
    if request.method == "POST":
        file1 = request.files["file1"]
        file2 = request.files["file2"]

        if file1.filename == "" or file2.filename == "" or not allowed_file(file1.filename) or not allowed_file(file2.filename):
            return redirect(url_for("comparator"))

        filename1 = str(uuid.uuid4()) + os.path.splitext(file1.filename)[1]
        filename2 = str(uuid.uuid4()) + os.path.splitext(file2.filename)[1]
        file_path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file1.save(file_path1)
        file2.save(file_path2)

        metadata1 = get_image_metadata(file_path1)
        metadata2 = get_image_metadata(file_path2)

        image1 = Image.open(file_path1).convert("RGB")
        image2 = Image.open(file_path2).convert("RGB")

        image1_tensor = transform(image1).unsqueeze(0).to(device)
        image2_tensor = transform(image2).unsqueeze(0).to(device)

        with torch.no_grad():
            output1 = model(image1_tensor)
            output2 = model(image2_tensor)
            probabilities1 = F.softmax(output1, dim=1)
            probabilities2 = F.softmax(output2, dim=1)
            confidence1, predicted_class1 = torch.max(probabilities1, 1)
            confidence2, predicted_class2 = torch.max(probabilities2, 1)
            confidence1 = confidence1.item() * 100
            confidence2 = confidence2.item() * 100

            if predicted_class1.item() == 1:
                prediction_label1 = "FAKE"
            else:
                prediction_label1 = "REAL"

            if predicted_class2.item() == 1:
                prediction_label2 = "FAKE"
            else:
                prediction_label2 = "REAL"

        return render_template("comparator.html", filename1=filename1, filename2=filename2, prediction1=prediction_label1, prediction2=prediction_label2, confidence1=f"{confidence1:.2f}", confidence2=f"{confidence2:.2f}", metadata1=metadata1, metadata2=metadata2)

    return render_template("comparator.html")

# /upload and /test should alwways be in order as declared below, add more routes above this comment
@app.route("/upload", methods=["GET", "POST"])
def upload():
    return render_template("upload.html")
#get_image_metadata JATIN KA CODE HAI POORA
def get_image_metadata(image_path):
    metadata={}
    
    metadata["File Format"]=os.path.splitext(image_path)[1].upper().replace(".","")
    
    file_size=os.path.getsize(image_path)
    metadata["File Size"]=f"{file_size/1024:.2f} KB"
    
    sha256_hash=hashlib.sha256()
    with open(image_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    metadata["Hash Value"]=sha256_hash.hexdigest()
    
    
    with open(image_path, "rb") as f:
        tags=exifread.process_file(f)
        if "EXIF DateTimeOriginal" in tags:
            date_created=tags["EXIF DateTimeOriginal"].values
            metadata["Date Created"]=date_created
        else:
            metadata["Date Created"]="No Created Date and Time available in metadata"
    
    
    return metadata

def get_video_metadata(video_path):
    metadata={}
    
    metadata["File Format"]=os.path.splitext(video_path)[1].upper().replace(".","")
    metadata["File Name"] = os.path.basename(video_path)

    file_size=os.path.getsize(video_path)
    metadata["File Size"]=f"{file_size/1024:.2f} KB"
    
    sha256_hash=hashlib.sha256()
    with open(video_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    metadata["Hash Value"]=sha256_hash.hexdigest()
    
    upload_time=os.path.getctime(video_path)
    metadata["Upload Time"]=datetime.fromtimestamp(upload_time).strftime('%d-%m-%Y %I:%M:%p')
    
    video=None
    try:
        video= VideoFileClip(video_path)
        metadata["Video Duration"]=f"{video.duration:.2f} seconds"
    except Exception as e:
        metadata["Video Duration"]="Video Duration is not available"
    finally:
        if video:
            video.close()
        
    return metadata
# @app.route("/test", methods=["POST"])
# def test():
    if "files" not in request.files:
        return redirect(url_for("upload"))

    file = request.files["files"]

    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("upload"))

    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    metadata=get_image_metadata(file_path)
    metadata["File Name"]=file.filename
    print(f"Metadata: {metadata}")

    image = Image.open(file_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        confidence = confidence.item() * 100  # Convert to percentage
        if predicted_class.item() == 1:
            prediction_label = "FAKE"
        else:
            prediction_label = "REAL"
        prediction = f" {prediction_label}"

    new_image = ImageModel(filename=filename, prediction=prediction, image_metadata=str(metadata))
    db.session.add(new_image)
    db.session.commit()

    images = ImageModel.query.all()
    report_path = report_pdf(filename, prediction)

    return render_template("image.html", prediction=prediction, metadata=metadata, filename=filename, report_filename=f"Report on {filename}.pdf", confidence=f"{confidence:.2f}")


@app.route("/test", methods=["POST"])
def test():
    if "files" not in request.files:
        return redirect(url_for("upload"))

    file = request.files["files"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("upload"))

    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    metadata = get_image_metadata(file_path)
    metadata["File Name"] = file.filename

    image = Image.open(file_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    image_tensor.requires_grad_(True)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        confidence = confidence.item() * 100

        if predicted_class.item() == 1:
            prediction_label = "FAKE"
        else:
            prediction_label = "REAL"
        prediction = f" {prediction_label}"

    heatmap_path = None
    if prediction_label == "FAKE":
        try:
            # Generate heatmap
            heatmap = apply_grad_cam(model, image_tensor)
            print("Heatmap generated:", heatmap.shape)  # Debugging

            # Resize heatmap to match the original image dimensions
            heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
            print("Resized heatmap:", heatmap.shape)  # Debugging

            # Convert heatmap to a color map
            heatmap = np.uint8(255 * heatmap)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Superimpose heatmap on the original image
            superimposed_img = cv2.addWeighted(np.array(image), 0.6, heatmap_colored, 0.4, 0)
            superimposed_img = Image.fromarray(superimposed_img)

            # Save the heatmap image
            heatmap_filename = f"heatmap_{filename}"
            heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
            superimposed_img.save(heatmap_path)
            print("Heatmap saved at:", heatmap_path)  # Debugging
        except Exception as e:
            print("Error generating heatmap:", str(e))  # Debugging

    new_image = ImageModel(filename=filename, prediction=prediction_label, image_metadata=str(metadata))
    db.session.add(new_image)
    db.session.commit()

    # Generate the report
    report_filename = f"Report on {filename}.pdf"
    report_path = report_pdf(filename, prediction, metadata)

    return render_template("image.html", 
                           prediction=prediction_label, 
                           metadata=metadata, 
                           filename=filename, 
                           heatmap_url=heatmap_filename if heatmap_path else None, 
                           confidence=f"{confidence:.2f}", 
                           report_filename=report_filename) 

#/videoupload and /test_vid will be in the same order declared below, shall not be altered 
@app.route("/videoupload", methods=["GET", "POST"])
def videoupload():
    return render_template("videoupload.html")

@app.route("/test_vid", methods=["POST"])
def test_vid():
    if "video" not in request.files:
        return redirect(url_for('test_vid'))
    
    file = request.files["video"]

    if file.filename == "" or not file.filename.lower().endswith(('.mp4', '.mkv', '.avi')):
        return redirect(url_for('test_vid'))
    
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    metadata=get_video_metadata(file_path)
    metadata["File Name"]=file.filename
    print(f"Video Metadata: {metadata}")

    frames = frame_extraction(file_path)
    frame_scores = []

    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model0(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            frame_scores.append((predicted_class.item(), confidence.item()))

    os.remove(file_path)

    if not frame_scores:
        return render_template("video.html", prediction="No frames processed. Try another video.")

    fake_count = sum(1 for cls, _ in frame_scores if cls == 1)
    real_count = len(frame_scores) - fake_count
    avg_confidence = sum(conf for _, conf in frame_scores) / len(frame_scores)

    if fake_count > real_count:
        final_prediction = "FAKE"
    else:
        final_prediction = "REAL"

    prediction = f"{final_prediction}"

    new_image = VideoModel(filename=filename, prediction=f"{final_prediction}", confidence=f"{avg_confidence*100:.2f}", image_metadata=str(metadata))
    db.session.add(new_image)
    db.session.commit()

    images = VideoModel.query.all()

    report_path0 = report_pdf0(filename, prediction, fake_count, real_count, avg_confidence,metadata)

    return render_template("video.html", prediction=f"{final_prediction}", report_filename0=f"Report on {filename}.pdf", confidence=f"{avg_confidence*100:.2f}",metadata=metadata, filename=filename)

@app.route("/detail-image/<filename>", methods=["GET"])
def detail_image(filename):
    image = ImageModel.query.filter_by(filename=filename).first()
    if image:
        metadata = eval(image.image_metadata)
        return render_template("detail-image.html", filename=filename, metadata=metadata, prediction=image.prediction, confidence=image.confidence)
    else:
        return "Image not found", 404

@app.route("/detail-video/<filename>", methods=["GET"])
def detail_video(filename):
    video = VideoModel.query.filter_by(filename=filename).first()
    if video:
        metadata = eval(video.image_metadata)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        frames = frame_extraction(file_path)
        frame_scores = []

        for frame in frames:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model0(image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                frame_scores.append((predicted_class.item(), confidence.item()))

        fake_count = sum(1 for cls, _ in frame_scores if cls == 1)
        real_count = len(frame_scores) - fake_count

        return render_template(
            "detail-video.html",
            filename=filename,
            metadata=metadata,
            prediction=video.prediction,
            confidence=video.confidence,
            fake_count=fake_count,  # Recalculated fake_count
            real_count=real_count   # Recalculated real_count
        )
    else:
        return "Video not found", 404

@app.route("/image", methods=["GET"])
def image():
    filename = request.args.get("filename", "")
    prediction = request.args.get("prediction", "No result available")
    return render_template("image.html", filename=filename, prediction=prediction)

@app.route("/video", methods=["GET"])
def video0():
    filename = request.args.get("filename", "")
    prediction = request.args.get("prediction", "No result available")
    video_path = os.path.join("static/uploads", filename)
    metadata = get_video_metadata(video_path) if os.path.exists(video_path) else {}
    return render_template("video.html", filename=filename, prediction=prediction, metadata=metadata)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route("/download0/<filename>")
def download0(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

with app.app_context():
    db.create_all()
    app.run(debug=True)