# Streamlit Deployment Fix for GMSH Dependency

## Problem
The app crashes with: `OSError: libGLU.so.1: cannot open shared object file: No such file or directory`

This happens because GMSH requires OpenGL libraries that aren't available by default in Streamlit Cloud.

## Solution

### Step 1: Add `packages.txt` to your repository

Create a file named `packages.txt` in the root of your repository with the following content:

```
libglu1-mesa
libgl1-mesa-glx
libxrender1
libxext6
libsm6
libxrandr2
```

### Step 2: Commit and push the file

```bash
git add packages.txt
git commit -m "Add system dependencies for GMSH"
git push
```

### Step 3: Redeploy on Streamlit Cloud

Once you push `packages.txt`, Streamlit Cloud will automatically:
1. Detect the file
2. Install the system packages using `apt-get`
3. Redeploy your app

## How it Works

Streamlit Cloud looks for a `packages.txt` file in your repository root. Each line in this file is treated as a package name to be installed via Ubuntu's `apt-get` package manager before your app starts.

The packages we're installing:
- `libglu1-mesa`: OpenGL Utility Library (the missing libGLU.so.1)
- `libgl1-mesa-glx`: OpenGL library
- `libxrender1`, `libxext6`, `libsm6`, `libxrandr2`: X11 dependencies needed by OpenGL

## Alternative: Use Headless GMSH

If the above doesn't work, you can also try initializing GMSH in headless mode by modifying your `test3.py`:

```python
def patch_gmsh():
    """Fix GMSH signal handling and run headless"""
    import signal as sig
    import os
    
    # Set headless mode
    os.environ['GMSH_OPTIONS'] = '-display :0.0'
    
    orig = sig.signal
    def dummy(sn, h):
        try:
            return orig(sn, h)
        except ValueError:
            return None
    sig.signal = dummy
    return orig
```

## Verification

After deploying, check the Streamlit Cloud logs. You should see lines like:
```
Reading package lists...
Building dependency tree...
Reading state information...
The following NEW packages will be installed:
  libglu1-mesa ...
```

If you see these messages, the system packages are being installed correctly.

## Additional Notes

- The `packages.txt` file must be in the repository root (same level as your Python files)
- Package names must be valid Ubuntu/Debian package names
- Changes take effect on the next deployment
- You can check available packages at: https://packages.ubuntu.com/

## Troubleshooting

If the app still fails after adding `packages.txt`:

1. Check Streamlit Cloud logs for package installation errors
2. Verify the package names are correct for the Ubuntu version Streamlit Cloud uses
3. Try adding additional dependencies if error messages mention other missing libraries
4. Consider using `apt-file` locally to find which package provides a missing `.so` file

## Files to Upload

Upload both of these files to your GitHub repository:
1. `packages.txt` (in repository root)
2. Your existing `test4.py`, `test3.py`, and `requirements.txt` files
