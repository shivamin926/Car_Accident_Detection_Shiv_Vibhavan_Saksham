import webbrowser
import subprocess
import time

# URL for your front-end HTML file
# Change the path depending on where index.html is
frontend_path = r"file:///C:/Users/shiva/OneDrive/Desktop/School/Machine Learning/Final_Project/Car_Accident_Detection_Shiv_Vibhavan_Saksham/client/index.html"

# Start uvicorn
process = subprocess.Popen(
    ["uvicorn", "backend.entrypoint:app", "--reload", "--port", "8000"]
)

# Wait a moment for the server to start
time.sleep(2)

# Open your browser
webbrowser.open(frontend_path)

# Keep the script running so uvicorn doesn't close
process.wait()
