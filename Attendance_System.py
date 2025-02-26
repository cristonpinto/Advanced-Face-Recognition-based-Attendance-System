import streamlit as st
import streamlit.components.v1 as components
import face_recognition
from datetime import datetime
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
import time

# Page configuration
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern look
st.markdown("""
<style>
    /* Modern color palette */
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --secondary: #64748b;
        --accent: #0ea5e9;
        --background: #f8fafc;
        --surface: #1e293b;
        --text: #0f172a;
        --success: #059669;
        --error: #dc2626;
    }

    /* Main layout styles */
    .main {
        background-color: var(--background);
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styles */
    .header {
        padding: 2rem 0;
        text-align: center;
        color: var(--primary);
        margin-bottom: 2.5rem;
        border-bottom: 1px solid #e2e8f0;
        background: linear-gradient(to right, #2563eb0d, #2563eb1a, #2563eb0d);
    }
    
    /* Card styles */
    .card {
        background-color: var(--surface);
        border-radius: 1rem;
        padding: 1.75rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    }
    
    /* Button styles */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 0.5rem;
        padding: 0.625rem 1.25rem;
        font-weight: 600;
        transition: all 0.2s ease;
        border: none;
        outline: none;
    }
    
    .stButton>button:hover {
        background-color: var(--primary-dark);
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgb(37 99 235 / 0.2);
    }
    
    .stButton>button:active {
        transform: translateY(0px);
    }
    
    /* Sidebar styles */
    .css-1d391kg {
        background-color: var(--primary);
        background-image: linear-gradient(135deg, var(--primary), var(--primary-dark));
    }
    
    .sidebar .sidebar-content {
        background-color: var(--primary);
        background-image: linear-gradient(135deg, var(--primary), var(--primary-dark));
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: var(--success);
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: .7;
            transform: scale(1.1);
        }
    }
    
    /* Table styles */
    .dataframe {
        border-collapse: separate;
        border-spacing: 0;
        margin: 2rem 0;
        font-size: 0.875rem;
        font-family: 'Inter', sans-serif;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        border-radius: 0.75rem;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    
    .dataframe thead tr {
        background-image: linear-gradient(to right, var(--primary), var(--primary-dark));
        color: white;
        text-align: left;
        font-weight: 600;
    }
    
    .dataframe th,
    .dataframe td {
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .dataframe tbody tr {
        transition: background-color 0.2s ease;
    }

    .dataframe tbody tr:nth-of-type(even) {
        background-color: #f8fafc;
    }

    .dataframe tbody tr:hover {
        background-color: #f1f5f9;
    }

    .dataframe tbody tr:last-of-type td {
        border-bottom: none;
    }

    /* Input fields */
    .stTextInput>div>div>input {
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
        padding: 0.625rem 1rem;
        transition: all 0.2s ease;
    }

    .stTextInput>div>div>input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgb(37 99 235 / 0.1);
    }

    /* Select boxes */
    .stSelectbox>div>div>div {
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }

    /* Progress bars */
    .stProgress>div>div>div>div {
        background-color: var(--primary);
        background-image: linear-gradient(to right, var(--primary), var(--accent));
        border-radius: 1rem;
    }

    /* Alerts and notifications */
    .stAlert {
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar menu with modern icons
st.sidebar.markdown('<div style="text-align: center; margin-bottom: 20px;"><h1 style="color: white;">📋 Menu</h1></div>', unsafe_allow_html=True)

menu = ["🏠 Home", "📸 Mark Attendance", "👤 Register", "📊 Attendance Sheet", "ℹ️ Help"]
choice = st.sidebar.selectbox("", menu)

# Initialize frame window for camera
FRAME_WINDOW = st.empty()

# Paths
path = 'Register_Data'
images = []
classNames = []

# Create Register_Data directory if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)
    
# Create Attendance_Sheet.csv if it doesn't exist
if not os.path.exists('Attendance_Sheet.csv'):
    with open('Attendance_Sheet.csv', 'w') as f:
        f.write('NAME,TIME,DATE')

# Check if Register_Data exists and has files
if os.path.exists(path):
    myList = os.listdir(path)
else:
    myList = []

# Main content based on menu selection
if choice == "🏠 Home":
    # Home page
    st.markdown('<div class="header"><h1>Welcome to Advanced Face Recognition Attendance System</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h2>Smart Attendance Made Simple</h2>
            <p>This advanced system uses facial recognition technology to automate attendance tracking, providing:</p>
            <ul>
                <li>Contactless attendance marking</li>
                <li>Real-time recognition and verification</li>
                <li>Secure and accurate record keeping</li>
                <li>Easy registration process</li>
            </ul>
            <p>Get started by registering your face using the Register option in the sidebar!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System status
        st.markdown("""
        <div class="card">
            <h3>System Status</h3>
            <p><span class="status-indicator status-active"></span> System active and ready</p>
            <p>Total registered users: <strong>{}</strong></p>
        </div>
        """.format(len(myList)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h3>Quick Navigation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("➕ Register New User"):
            st.session_state['menu_selection'] = "👤 Register"
            st.experimental_rerun()
        
        if st.button("📸 Mark Attendance"):
            st.session_state['menu_selection'] = "📸 Mark Attendance"
            st.rerun()
        
        if st.button("📊 View Attendance Records"):
            st.session_state['menu_selection'] = "📊 Attendance Sheet"
            st.experimental_rerun()

elif choice == "📸 Mark Attendance":
    st.markdown('<div class="header"><h1>Mark Attendance</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        FRAME_WINDOW = st.image([])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Attendance Controls")
        run = st.checkbox("Start Camera")
        
        if run:
            st.markdown('<p><span class="status-indicator status-active"></span> Camera active</p>', unsafe_allow_html=True)
            st.info("Stand in front of the camera. Your attendance will be marked automatically when your face is recognized.")
        else:
            st.warning("Click the checkbox to activate camera and mark attendance.")
        
        if st.button("Stop and Reset"):
            run = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent activity
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Recent Activity")
        try:
            df = pd.read_csv('Attendance_Sheet.csv')
            if len(df) > 0:
                st.write(df.tail(3))
            else:
                st.write("No records yet.")
        except:
            st.write("No records yet.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if run:
        # Load registered faces
        for cl in myList:
            curlImg = cv2.imread(f'{path}/{cl}')
            images.append(curlImg)
            classNames.append(os.path.splitext(cl)[0])
        
        def findEncodings(images):
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                try:
                    encode = face_recognition.face_encodings(img)[0]
                    encodeList.append(encode)
                except:
                    st.error(f"No face found in one of the images. Please check your registered images.")
            return encodeList
        
        def markAttendance(name):
            with open('Attendance_Sheet.csv', 'r+') as f:
                myDataList = f.readlines()
                nameList = []
                for line in myDataList:
                    entry = line.split(',')
                    nameList.append(entry[0])
                if name not in nameList:
                    now = datetime.now()
                    dtString = now.strftime('%H:%M:%S')
                    dStr = now.strftime('%d:%m:%Y')
                    f.writelines(f'\n{name},{dtString},{dStr}')
                    return True
                return False
        
        if images:
            with st.spinner("Loading facial recognition model..."):
                encodeListKnown = findEncodings(images)
                st.success("Facial recognition model loaded successfully!")
            
            cap = cv2.VideoCapture(0)
            
            # Set lower resolution for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            attendance_marked = False
            
            while run:
                success, img = cap.read()
                if not success:
                    st.error("Failed to access camera. Please check your camera connection.")
                    break
                
                # Resize for faster processing
                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                
                faceCurFrame = face_recognition.face_locations(imgS)
                encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
                
                # Add timestamp to image
                now = datetime.now()
                timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
                cv2.putText(img, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    
                    if len(faceDis) > 0:
                        matchIndex = np.argmin(faceDis)
                        
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                        
                        if matches[matchIndex]:
                            name = classNames[matchIndex].upper()
                            # Create a nicer looking rectangle
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add a background for the name
                            cv2.rectangle(img, (x1, y2-40), (x2, y2), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, name, (x1+6, y2-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                            
                            # Show confidence
                            confidence = round((1 - faceDis[matchIndex]) * 100, 2)
                            cv2.putText(img, f"Conf: {confidence}%", (x1+6, y2-65), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                            
                            # Mark attendance
                            if confidence > 50:  # Only mark if confidence is good
                                is_new = markAttendance(name)
                                if is_new:
                                    # Add a "Marked" indicator
                                    cv2.putText(img, "ATTENDANCE MARKED", (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                                    attendance_marked = True
                        else:
                            # Unknown face
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.rectangle(img, (x1, y2-40), (x2, y2), (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, "Unknown", (x1+6, y2-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                
                # Convert to RGB for display
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(img)
                
                # If attendance was marked, pause briefly
                if attendance_marked:
                    time.sleep(2)
                    attendance_marked = False
                
                # Add a small delay to reduce CPU usage
                time.sleep(0.01)
            
            # Release the camera when done
            cap.release()
        else:
            st.warning("No registered faces found. Please register at least one face first.")
            if st.button("Go to Registration"):
                st.session_state['menu_selection'] = "👤 Register"
                st.experimental_rerun()

elif choice == "👤 Register":
    st.markdown('<div class="header"><h1>Register New User</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Upload Photo for Registration")
        
        # Name input
        name = st.text_input("Full Name (This will appear in attendance records)")
        
        # File uploader with instructions
        st.markdown("""
        <p>Upload a clear facial photo:</p>
        <ul>
            <li>Face should be clearly visible</li>
            <li>Good lighting conditions</li>
            <li>Neutral expression recommended</li>
            <li>Supported formats: JPG, JPEG, PNG</li>
        </ul>
        """, unsafe_allow_html=True)
        
        image_file = st.file_uploader("", type=['png', 'jpeg', 'jpg'])
        
        if image_file is not None:
            # Preview the image
            img = Image.open(image_file)
            st.image(img, width=300, caption="Preview")
            
            if name:
                # Save with the name instead of original filename
                file_extension = os.path.splitext(image_file.name)[1]
                new_filename = f"{name}{file_extension}"
                
                # Submit button
                if st.button("Register User"):
                    with open(os.path.join(path, new_filename), "wb") as f:
                        f.write(image_file.getbuffer())
                    st.success(f"✅ Successfully registered {name}!")
                    
                    # Show a "Mark attendance now" button
                    if st.button("Mark Attendance Now"):
                        st.session_state['menu_selection'] = "📸 Mark Attendance"
                        st.rerun()
            else:
                st.warning("Please enter your name before registering.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Registration Guidelines")
        st.markdown("""
        For best results:
        
        1. Use a recent photo
        2. Ensure your face is clearly visible
        3. Avoid wearing sunglasses or hats
        4. Use proper lighting
        5. Look directly at the camera
        
        The system will use this image to identify you when marking attendance.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show registered users
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Registered Users")
        
        if myList:
            for user in myList:
                username = os.path.splitext(user)[0]
                st.markdown(f"• {username}")
        else:
            st.info("No users registered yet.")
        st.markdown('</div>', unsafe_allow_html=True)

elif choice == "📊 Attendance Sheet":
    st.markdown('<div class="header"><h1>Attendance Records</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Attendance Sheet")
        
        try:
            df = pd.read_csv('Attendance_Sheet.csv')
            
            # Data cleaning and enhancement
            if len(df) > 0:
                # Rename columns for better display
                df.columns = ['Name', 'Time', 'Date']
                
                # Format the data
                if 'Time' in df.columns:
                    df['Time'] = df['Time'].astype(str)
                if 'Date' in df.columns:
                    df['Date'] = df['Date'].astype(str)
                
                # Add a status column (just for visual enhancement)
                df['Status'] = 'Present'
                
                # Provide filter options
                name_filter = st.selectbox("Filter by Name", ['All'] + list(df['Name'].unique()))
                date_filter = st.selectbox("Filter by Date", ['All'] + list(df['Date'].unique()))
                
                filtered_df = df.copy()
                if name_filter != 'All':
                    filtered_df = filtered_df[filtered_df['Name'] == name_filter]
                if date_filter != 'All':
                    filtered_df = filtered_df[filtered_df['Date'] == date_filter]
                
                # Display the dataframe with some styling
                st.dataframe(filtered_df, use_container_width=True)
                
                # Download options
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name='attendance_records.csv',
                    mime='text/csv',
                )
            else:
                st.info("No attendance records found.")
        except Exception as e:
            st.error(f"Error reading attendance sheet: {e}")
            if not os.path.exists('Attendance_Sheet.csv'):
                # Create the file if it doesn't exist
                with open('Attendance_Sheet.csv', 'w') as f:
                    f.write('NAME,TIME,DATE')
                st.info("Created a new attendance sheet. No records yet.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Attendance Statistics")
        
        try:
            df = pd.read_csv('Attendance_Sheet.csv')
            if len(df) > 0:
                # Rename columns
                df.columns = ['Name', 'Time', 'Date']
                
                # Display statistics
                st.write(f"Total Records: {len(df)}")
                st.write(f"Unique Users: {df['Name'].nunique()}")
                
                # Latest attendance
                st.subheader("Latest Attendance")
                st.write(df.tail(5)[['Name', 'Time', 'Date']])
            else:
                st.info("No data available for statistics.")
        except:
            st.info("No data available for statistics.")
        st.markdown('</div>', unsafe_allow_html=True)

elif choice == "ℹ️ Help":
    st.markdown('<div class="header"><h1>Help & Instructions</h1></div>', unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Getting Started", "FAQ", "Troubleshooting"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("How to Use This System")
        
        st.markdown("""
        ### Step 1: Registration
        - Go to the **Register** section from the sidebar menu
        - Enter your full name
        - Upload a clear photo of your face
        - Click "Register User"
        
        ### Step 2: Mark Attendance
        - Navigate to the **Mark Attendance** section
        - Check the "Start Camera" box to activate your camera
        - Position yourself in front of the camera
        - The system will automatically recognize your face and mark attendance
        - A green box around your face indicates successful recognition
        
        ### Step 3: View Records
        - Go to the **Attendance Sheet** section to view all attendance records
        - Filter by name or date as needed
        - Download records in CSV format for reporting
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Frequently Asked Questions")
        
        # FAQ as expandable sections
        with st.expander("Can I register multiple people?"):
            st.write("Yes, you can register as many users as needed. Each person should have their own registration with a clear facial photo.")
            
        with st.expander("Can I mark attendance multiple times in a day?"):
            st.write("Currently, the system is designed to record attendance once per day. If you need to mark both entry and exit times, please contact the system administrator.")
            
        with st.expander("What if the system doesn't recognize my face?"):
            st.write("""
            Try the following:
            - Ensure there's good lighting on your face
            - Remove glasses or hats if possible
            - Position yourself directly in front of the camera
            - Re-register with a clearer photo if needed
            """)
            
        with st.expander("Is my data secure?"):
            st.write("Yes, all facial recognition data is stored locally and not shared with any external services.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Troubleshooting")
        
        st.markdown("""
        ### Camera Issues
        - Ensure your camera is properly connected
        - Check if other applications are using your camera
        - Try restarting the application
        
        ### Registration Problems
        - Ensure your photo has a clear view of your face
        - Use good lighting conditions
        - Make sure the file is in JPG, JPEG, or PNG format
        
        ### Recognition Failures
        - Try registering with a new, clearer photo
        - Ensure proper lighting when marking attendance
        - Position yourself directly facing the camera
        
        ### CSV File Issues
        - Do not manually edit the Attendance_Sheet.csv file
        - If the file becomes corrupted, you can delete it and a new one will be created automatically
        
        For additional help, please contact system support.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Add a simple footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #6b7280; font-size: 0.8em;">
    Advanced Face Recognition Attendance System © 2025
</div>
""", unsafe_allow_html=True)