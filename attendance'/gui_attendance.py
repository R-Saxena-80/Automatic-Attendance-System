# GUI-based Automatic Attendance System using Tkinter + DeepFace
# This GUI wraps your existing attendance code logic

import tkinter as tk
from tkinter import messagebox, simpledialog
import threading

# ---- IMPORT YOUR EXISTING FUNCTIONS ----
# Make sure this file is in the SAME directory as your original script
from attendance_system import (
    capture_images_for_student,
    build_embeddings,
    mark_attendance
)

# ---------------- GUI APP ----------------
class AttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Automatic Attendance System")
        self.root.geometry("420x350")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.safe_exit)


        title = tk.Label(root, text="Automatic Attendance System",
                         font=("Helvetica", 16, "bold"))
        title.pack(pady=20)

        btn_add = tk.Button(root, text="Add New Student",
                            width=25, height=2,
                            command=self.add_student)
        btn_add.pack(pady=10)

        btn_embed = tk.Button(root, text="Build Embeddings",
                              width=25, height=2,
                              command=self.build_embeddings)
        btn_embed.pack(pady=10)

        btn_att = tk.Button(root, text="Start Attendance",
                            width=25, height=2,
                            command=self.start_attendance)
        btn_att.pack(pady=10)

        btn_exit = tk.Button(root, text="Exit",
                     width=25, height=2,
                     command=self.safe_exit)

        btn_exit.pack(pady=20)

    # ---------- Button Actions ----------
    def add_student(self):
        name = simpledialog.askstring("Student Name", "Enter student name:")
        if not name:
            return
        self.run_thread(capture_images_for_student, name)

    def build_embeddings(self):
        self.run_thread(build_embeddings)

    def start_attendance(self):
        messagebox.showinfo("Info",
                            "Press 'q' in camera window to stop attendance")
        self.run_thread(mark_attendance)

    # ---------- Thread Helper ----------
    def run_thread(self, func, *args):
        t = threading.Thread(target=self.safe_run, args=(func, *args))
        t.daemon = True
        t.start()

    def safe_run(self, func, *args):
        try:
            func(*args)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def safe_exit(self):
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            try:
                self.root.quit()     # Stop mainloop
                self.root.destroy()  # Close window
            except Exception:
                pass



# ---------------- MAIN ----------------
if __name__ == '__main__':
    root = tk.Tk()
    app = AttendanceGUI(root)
    root.mainloop()

"""
INSTRUCTIONS:
1. Rename your original file to: attendance_system.py
2. Save this file as: gui_attendance.py
3. Run using: python gui_attendance.py

WHY THREADING IS USED:
- Prevents GUI freezing when camera or DeepFace is running

OPTIONAL ENHANCEMENTS:
- Show live camera inside Tkinter
- Student list viewer
- Attendance history viewer (CSV)
- Admin login system
"""
