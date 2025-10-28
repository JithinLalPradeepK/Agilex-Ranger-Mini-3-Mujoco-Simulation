# Agilex-Ranger-Mini-3-Mujoco-Simulation

Simulation of the **Agilex Ranger Mini 3** mobile robot in **MuJoCo**. This project includes various control methods for the robot:
* **PID Controller** for Angle and Velocity control.
* **Open-loop control** for Ackerman, Oblique, and Spin Steering.
* **Trajectory Tracking** for both discrete points and smooth curves.

**System Requirement:** This simulation was developed and tested on **Windows 10**.

---

## 💻 Installation and Setup

This guide details the installation of the **Python version of MuJoCo 2.2.1** and the configuration of the development environment using **VS Code**.

### A) Installing Python3 on Windows (Skip if already installed)

1.  **Check for Python:** Open a **command prompt** (`cmd`) and type `python3`.
    * If installed, you'll see the version number and a prompt (`>>>`). Test it with commands like `print(3+2)`.
    * To quit, type `quit()`.
2.  **Download and Install:** If not installed, download the latest version from [https://www.python.org/downloads/](https://www.python.org/downloads/).
    * Run the downloaded `.exe` file to install.
    * Once done, check the installation by repeating step 1.

---

### B) Installing VS Code and Python Extension (Skip if already installed)

1.  **Install VS Code:** Download and run the installer from [https://code.visualstudio.com/](https://code.visualstudio.com/).
2.  **Install Python Extension:**
    * Open **VS Code**.
    * Click on the **Extensions** view icon (or press `Ctrl+Shift+X`).
    * Search for **`Python`** by Microsoft.
    * Click **Install**. This extension provides IntelliSense, debugging, and code-running capabilities.
3.  **Test VS Code Setup:**
    * Go to **File > New Text File**.
    * Type `print(2+3)`.
    * Save the file as `test.py`.
    * Run the file by clicking the **Run Python File** button ($\blacktriangleright$) in the top-right corner, or right-click and select **Run Python File in Terminal**.
    * The result should appear in the **Terminal** window.

---

### C) Installing MuJoCo Python Package and Running the Simulation

1.  **Install MuJoCo and Scipy:**
    * Open the terminal (or command prompt) and run the following command(s).

        ```bash
        pip install mujoco
        pip install scipy
        ```

2.  **Clone the Repository:**
    * Clone this entire project directory to your local workspace.
3.  **Run the Simulation:**
    * Open any of the python files in **VS Code**.
    * Run the python file required to start the Ranger Mini simulation.
    * To edit the control logic, change the navigation controller function.


---

## 📚 Resources

For a detailed video walkthrough of the MuJoCo installation:

* **MuJoCo Detailed Installation:** [https://www.youtube.com/watch?v=tMo2zyaNCDQ&list=PLc7bpbeTIk75dgBVd07z6\_uKN1KQkwFRK&index=2](https://www.youtube.com/watch?v=tMo2zyaNCDQ&list=PLc7bpbeTIk75dgBVd07z6_uKN1KQkwFRK&index=2) 
