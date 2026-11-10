# ============================================================
# 01_install_wsl.ps1
# Run this script in an ELEVATED (Run as Administrator)
# PowerShell session on Windows.
#
# What it does:
#   1. Enables the WSL optional feature
#   2. Enables the Virtual Machine Platform feature
#   3. Sets WSL 2 as the default version
#   4. Installs the Ubuntu-24.04 distribution
#   5. Prints next-step instructions
#
# A REBOOT may be required between steps 1-2 and step 4.
# If the script exits asking you to reboot, do so and
# re-run the script; it will skip already-done steps.
# ============================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n[STEP] $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "  [OK] $msg"  -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  [!!] $msg"  -ForegroundColor Yellow }

# --- Require admin ---
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Please run this script as Administrator (right-click PowerShell > Run as Administrator)."
}

# --- 1. Enable WSL feature ---
Write-Step "Enabling Microsoft-Windows-Subsystem-Linux..."
$wslFeature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
if ($wslFeature.State -ne "Enabled") {
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart | Out-Null
    Write-OK "Feature enabled."
} else {
    Write-OK "Already enabled."
}

# --- 2. Enable Virtual Machine Platform ---
Write-Step "Enabling VirtualMachinePlatform..."
$vmFeature = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
if ($vmFeature.State -ne "Enabled") {
    Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart | Out-Null
    Write-OK "Feature enabled. A REBOOT is required before continuing."
    Write-Warn "After rebooting, re-run this script to continue from where it left off."
    Read-Host "Press Enter to reboot now, or Ctrl+C to reboot manually"
    Restart-Computer -Force
    exit
} else {
    Write-OK "Already enabled."
}

# --- 3. Set WSL 2 as default ---
Write-Step "Setting WSL default version to 2..."
wsl --set-default-version 2 | Out-Null
Write-OK "Default version set to 2."

# --- 4. Install Ubuntu-24.04 ---
Write-Step "Checking for Ubuntu-24.04..."
$installed = wsl --list --quiet 2>$null | Where-Object { $_ -match "Ubuntu-24.04" }
if (-not $installed) {
    Write-Step "Installing Ubuntu-24.04 (this downloads ~500 MB, please wait)..."
    wsl --install -d Ubuntu-24.04
    Write-OK "Ubuntu-24.04 installed. Follow the on-screen prompts to create a UNIX user."
} else {
    Write-OK "Ubuntu-24.04 already installed."
}

# --- 5. Update WSL kernel ---
Write-Step "Updating WSL kernel..."
wsl --update | Out-Null
Write-OK "WSL kernel up to date."

# --- Done ---
Write-Host @"

===========================================================
 WSL2 setup complete.

 NEXT STEPS:
   1. Open Ubuntu-24.04 from the Start Menu (or run: wsl -d Ubuntu-24.04)
   2. Confirm NVIDIA GPU is visible inside WSL:
        nvidia-smi
   3. If nvidia-smi is missing inside WSL, ensure the NVIDIA
      Windows driver (>= 555) is installed on the host.
      Driver 610.88 detected on this machine -- you are OK.
   4. cd into the project via WSL:
        cd /mnt/c/Users/Joseph\ Raymond/Documents/projects/model_run
   5. Run the next setup script:
        bash setup/02_setup_env.sh
===========================================================
"@ -ForegroundColor Green
