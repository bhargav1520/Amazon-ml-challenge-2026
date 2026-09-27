# AWS High-Performance Cloud Execution Guide (0.99+ Score Target)

This guide provides the exact step-by-step instructions from logging into AWS to launching a high-core compute instance, executing the 4-stage Entity Resolution pipeline, and downloading the final `matching_results.tsv`.

---

## 1. Top 10 AWS EC2 Instances (Prioritized Order)

| Priority | Instance Type | vCPUs | RAM | Architecture / Specs | Approx Price/hr | Estimated Total Runtime | Why Pick This |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **#1 (Best)** | **`c6i.16xlarge`** | **64** | **128 GB** | 3rd Gen Intel Xeon (Ice Lake) | ~$2.72 (Spot: ~$0.85) | **~10 – 12 mins** | **Top Choice**: Maximum CPU throughput for 12M+ pair extraction and LightGBM tree building. |
| **#2** | **`c6i.8xlarge`** | **32** | **64 GB** | 3rd Gen Intel Xeon | ~$1.36 (Spot: ~$0.45) | **~18 – 22 mins** | **Best Value**: Extremely fast and very cost-efficient. |
| **#3** | **`c5.18xlarge`** | **72** | **144 GB** | 2nd Gen Intel Xeon Scalable | ~$3.06 (Spot: ~$0.95) | **~10 – 13 mins** | High vCPU count for instant parallelized string feature processing. |
| **#4** | **`c6a.16xlarge`** | **64** | **128 GB** | 3rd Gen AMD EPYC (Milan) | ~$2.45 (Spot: ~$0.75) | **~11 – 14 mins** | High performance AMD compute at lower hourly rate. |
| **#5** | **`c5.9xlarge`** | **36** | **72 GB** | 2nd Gen Intel Xeon | ~$1.53 (Spot: ~$0.50) | **~20 – 25 mins** | Standard high-availability compute tier. |
| **#6** | **`m6i.16xlarge`** | **64** | **256 GB** | Intel Xeon (General Purpose) | ~$3.07 (Spot: ~$0.98) | **~11 – 13 mins** | Massive 256 GB RAM ensures 100% in-memory data processing without disk swapping. |
| **#7** | **`c6a.8xlarge`** | **32** | **64 GB** | 3rd Gen AMD EPYC | ~$1.22 (Spot: ~$0.40) | **~20 – 24 mins** | Budget-friendly 32-core AMD tier. |
| **#8** | **`m5.12xlarge`** | **48** | **192 GB** | Intel Xeon Platinum | ~$2.30 (Spot: ~$0.70) | **~15 – 18 mins** | Robust memory and compute balance. |
| **#9** | **`g4dn.4xlarge`** | **16** | **64 GB** | Intel + NVIDIA T4 (16GB GPU) | ~$1.20 (Spot: ~$0.40) | **~25 – 30 mins** | Good if GPU tree acceleration for CatBoost is desired. |
| **#10** | **`r6i.8xlarge`** | **32** | **256 GB** | High-Memory Optimized | ~$2.02 (Spot: ~$0.65) | **~22 – 26 mins** | Extremely safe against any out-of-memory risks. |

---

## 2. Step-by-Step AWS Setup & Navigation

### Step 2.1: Log into AWS & Navigate to EC2
1. Open [https://aws.amazon.com/console/](https://aws.amazon.com/console/) and click **Sign In to the Console**.
2. In the top search bar, type `EC2` and click **EC2 (Virtual Servers in the Cloud)**.
3. Check the top-right region selector and choose **US East (N. Virginia) `us-east-1`** or **Asia Pacific (Mumbai) `ap-south-1`** (recommended for low latency).

---

### Step 2.2: Launch the Instance (Tab-by-Tab Configuration)
1. On the EC2 Dashboard, click the orange **Launch Instance** button.
2. **Name and tags**:
   * Name: `amazon-ml-worker`
3. **Application and OS Images (Amazon Machine Image)**:
   * Select the **Ubuntu** tab.
   * Dropdown: **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type** (64-bit x86).
4. **Instance type**:
   * Click the dropdown and search for **`c6i.16xlarge`** (or `c6i.8xlarge` / `c5.9xlarge`).
5. **Key pair (login)**:
   * If you have a key pair, select it.
   * If not, click **Create new key pair**:
     * Key pair name: `amazon-ml-key`
     * Key pair type: `RSA`
     * Private key file format: `.pem` (for OpenSSH / Linux / macOS / Windows Terminal).
     * Click **Create key pair** (the `.pem` file will download to your computer).
6. **Network settings**:
   * Keep default VPC and Subnet.
   * Check **Auto-assign public IP: Enable**.
   * Firewall (security groups): Select **Create security group** $\rightarrow$ Check **Allow SSH traffic from: Anywhere (0.0.0.0/0)**.
7. **Configure storage (VERY IMPORTANT)**:
   * Change default `8 GiB` to **`150 GiB`**.
   * Volume type: **`gp3`** (3000 IOPS, 125 MB/s throughput).
8. Click **Launch instance** at the bottom right.
9. Click **View all instances** and wait until Instance State changes to **Running**.

---

## 3. Connecting to the Instance

Open your local terminal (Command Prompt, PowerShell, or macOS/Linux Terminal) in the folder where your `amazon-ml-key.pem` is downloaded:

### On macOS / Linux / WSL:
```bash
chmod 400 amazon-ml-key.pem
ssh -i "amazon-ml-key.pem" ubuntu@<YOUR-EC2-PUBLIC-IP>
```

### On Windows PowerShell:
```powershell
ssh -i "amazon-ml-key.pem" ubuntu@<YOUR-EC2-PUBLIC-IP>
```
*(Replace `<YOUR-EC2-PUBLIC-IP>` with the IPv4 Address shown in your AWS EC2 Console under "Public IPv4 address".)*

---

## 4. 1-Click Environment Setup on AWS

Once connected to your EC2 terminal, copy and paste this command block:

```bash
sudo apt update -y && sudo apt install -y python3-pip python3-venv git htop unzip
python3 -m venv env
source env/bin/activate

pip install --upgrade pip
pip install lightgbm catboost rapidfuzz joblib pyarrow fastparquet pandas numpy scikit-learn pytest
```

---

## 5. Clone Repository & Place Datasets

```bash
# 1. Clone the updated branch
git clone -b Bharath-solution https://github.com/bhargav1520/Amazon-ml-challenge-2026.git
cd Amazon-ml-challenge-2026/code/business_entity_resolution

# 2. Run unit tests to confirm 100% environment health
pytest -v
```

### Copy Datasets to EC2:
If you need to upload your `student_resource` folder from your local machine to EC2:

*From your local computer terminal:*
```bash
scp -i "amazon-ml-key.pem" -r student_resource ubuntu@<YOUR-EC2-PUBLIC-IP>:~/Amazon-ml-challenge-2026/
```

---

## 6. Execute the High-Capacity Pipeline

Run the automated runner with real-time logs:

```bash
python3 run_aws.py
```

### What this script automatically runs:
1. **Stage 1 (Preprocess)**: Cleans names and addresses, expands abbreviations, removes noisy legal terms, extracts number sets, and saves fast columnar Parquets.
2. **Stage 2 (Blocking)**: Builds 6-pass multi-indexer with `max_candidates=25` across 300k training entities and all test entities.
3. **Stage 3 (Training)**: Fits 800-tree dual LightGBM/CatBoost ensemble with 3x hard-negative mining and per-country threshold optimization.
4. **Stage 4 (Inference)**: Performs batched vector scoring over 1.73M test records with `max_matches_per_entity=15`.
5. **Validator**: Runs `validate_submission.py` to confirm format correctness.

---

## 7. Download Submission & Terminate Instance

### Step 7.1: Download `matching_results.tsv` to Your Local PC
*Run this on your local machine (not inside the SSH session):*

```bash
scp -i "amazon-ml-key.pem" ubuntu@<YOUR-EC2-PUBLIC-IP>:~/Amazon-ml-challenge-2026/code/business_entity_resolution/output/matching_results.tsv .
```

### Step 7.2: Terminate the Instance (Stop Billing)
1. Go back to the **AWS EC2 Console**.
2. Select your `amazon-ml-worker` instance.
3. Click **Instance state** $\rightarrow$ **Terminate instance**.
4. Confirm **Terminate**.
