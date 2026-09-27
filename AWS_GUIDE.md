# AWS High-Performance Cloud Execution Guide (0.99+ Score Target)

This guide provides exhaustive, step-by-step instructions for launching an AWS EC2 instance, setting up the environment, running the 4-stage Entity Resolution pipeline, and downloading the final `matching_results.tsv` submission file.

---

## 1. Top 15 AWS EC2 Instances (Prioritized for Student and Standard Accounts)

AWS student credit accounts often have vCPU quota limits (e.g., maximum 32 or 64 vCPUs, or specific instance families allowed). Choose the highest available instance type from this list:

| Priority | Instance Type | vCPUs | RAM | Specs / Family | Approx Price/hr | Estimated Total Runtime | Account Availability / Quota Fit |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **1** | **`c6i.16xlarge`** | **64** | **128 GB** | 3rd Gen Intel Xeon Ice Lake | ~$2.72 (Spot: ~$0.85) | **~10 – 12 mins** | High-quota accounts (64 vCPUs). Fastest compute. |
| **2** | **`c6i.8xlarge`** | **32** | **64 GB** | 3rd Gen Intel Xeon Ice Lake | ~$1.36 (Spot: ~$0.45) | **~18 – 22 mins** | Standard 32 vCPU student quota. Best overall value. |
| **3** | **`c5.18xlarge`** | **72** | **144 GB** | 2nd Gen Intel Xeon Scalable | ~$3.06 (Spot: ~$0.95) | **~10 – 13 mins** | High-quota accounts. Heavy CPU parallelization. |
| **4** | **`c6a.16xlarge`** | **64** | **128 GB** | 3rd Gen AMD EPYC (Milan) | ~$2.45 (Spot: ~$0.75) | **~11 – 14 mins** | Cost-effective AMD high-core tier. |
| **5** | **`c5.9xlarge`** | **36** | **72 GB** | 2nd Gen Intel Xeon | ~$1.53 (Spot: ~$0.50) | **~20 – 25 mins** | Standard 36 vCPU tier. Widely available. |
| **6** | **`m6i.16xlarge`** | **64** | **256 GB** | Intel Xeon General Purpose | ~$3.07 (Spot: ~$0.98) | **~11 – 13 mins** | 256 GB RAM keeps all indexes 100% in memory. |
| **7** | **`c6a.8xlarge`** | **32** | **64 GB** | 3rd Gen AMD EPYC | ~$1.22 (Spot: ~$0.40) | **~20 – 24 mins** | AMD 32-core instance within 32 vCPU limits. |
| **8** | **`m5.12xlarge`** | **48** | **192 GB** | Intel Xeon Platinum | ~$2.30 (Spot: ~$0.70) | **~15 – 18 mins** | Robust memory and core balance. |
| **9** | **`c5a.8xlarge`** | **32** | **64 GB** | 2nd Gen AMD EPYC | ~$1.23 (Spot: ~$0.38) | **~22 – 25 mins** | Very common on student tiers. |
| **10** | **`m6a.8xlarge`** | **32** | **128 GB** | 3rd Gen AMD EPYC | ~$1.38 (Spot: ~$0.45) | **~20 – 24 mins** | 128 GB RAM on a 32-core quota. |
| **11** | **`c5.4xlarge`** | **16** | **32 GB** | 2nd Gen Intel Xeon | ~$0.68 (Spot: ~$0.25) | **~35 – 40 mins** | Strict 16 vCPU student quota. |
| **12** | **`m5.4xlarge`** | **16** | **64 GB** | Intel Xeon Platinum | ~$0.77 (Spot: ~$0.27) | **~35 – 40 mins** | 64 GB RAM on 16 vCPUs. |
| **13** | **`c6i.4xlarge`** | **16** | **32 GB** | 3rd Gen Intel Xeon | ~$0.68 (Spot: ~$0.25) | **~30 – 35 mins** | High single-thread clock speed. |
| **14** | **`t3.2xlarge`** | **8** | **32 GB** | Burstable General Purpose | ~$0.33 (Spot: ~$0.11) | **~55 – 65 mins** | Guaranteed availability on all starter student accounts. |
| **15** | **`c5.2xlarge`** | **8** | **16 GB** | 2nd Gen Intel Xeon | ~$0.34 (Spot: ~$0.12) | **~60 – 70 mins** | Minimum compute tier for budget backup. |

---

## 2. Step-by-Step AWS Console Instructions

### Step 2.1: Sign In and Region Selection
1. Open your browser and go to: `https://aws.amazon.com/console/`
2. Click the orange **Sign In to the Console** button.
3. Enter your account credentials (IAM User or Root User).
4. Once inside the console, look at the top navigation bar on the far right (next to your username) for the **Region Selector**.
5. Click it and select **US East (N. Virginia) `us-east-1`** or **Asia Pacific (Mumbai) `ap-south-1`**.

---

### Step 2.2: Launch the EC2 Instance
1. In the top search bar, type `EC2` and press Enter. Click on the first service named **EC2**.
2. On the EC2 Dashboard, click the orange button named **Launch Instance**.
3. Fill in the following sections:

#### Section 1: Name and tags
* In the **Name** input field, type: `amazon-ml-worker`

#### Section 2: Application and OS Images (Amazon Machine Image)
* Click the **Ubuntu** icon box.
* Ensure the **Amazon Machine Image (AMI)** dropdown shows: **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type** (Architecture: 64-bit (x86)).

#### Section 3: Instance type
* Click the **Instance type** dropdown.
* Search for and select your chosen instance (e.g., `c6i.16xlarge`, `c6i.8xlarge`, or `c5.9xlarge`).
* *Note:* If AWS displays a message saying you have exceeded vCPU quota for that instance family, simply select the next instance down from the table in Section 1 (e.g., `c6i.8xlarge` or `c5.4xlarge`).

#### Section 4: Key pair (login)
* If you already have an existing `.pem` key pair, select it from the dropdown.
* If you do not have one, click **Create new key pair**:
  * Key pair name: `amazon-ml-key`
  * Key pair type: `RSA`
  * Private key file format: `.pem`
  * Click **Create key pair**. A file named `amazon-ml-key.pem` will automatically download to your computer. Keep this file safe.

#### Section 5: Network settings
* Click **Edit** on the right side of Network settings.
* Ensure **Auto-assign public IP** is set to **Enable**.
* Under **Firewall (security groups)**, select **Create security group**.
* Check the box for **Allow SSH traffic from** and keep it set to **Anywhere (0.0.0.0/0)**.

#### Section 6: Configure storage (CRITICAL)
* The default is `8 GiB`. You must change this to **`150 GiB`**.
* Volume type: **`gp3`**.
* *Reason:* 150 GiB gp3 gives 3,000 IOPS and 125 MB/s disk speed, preventing any out-of-disk errors during parquet indexing.

#### Section 7: Launch
* In the **Summary** panel on the right, verify your instance type and storage size (150 GiB).
* Click the orange **Launch instance** button.
* On the next page, click **View all instances** at the bottom right.
* In the instance table, find `amazon-ml-worker`. Wait 30 to 60 seconds until the **Instance state** column says **Running** (with a green circle).
* Click on `amazon-ml-worker`. Look in the bottom details panel and copy the **Public IPv4 address** (e.g., `54.210.88.120`).

---

## 3. Connecting to Your Instance via Terminal

Open a terminal on your computer.

* **On Windows**: Open PowerShell or Windows Terminal. Navigate to the folder where your `amazon-ml-key.pem` was downloaded (typically `cd ~/Downloads` or `cd C:\Users\<username>\Downloads`).
* **On macOS / Linux**: Open Terminal and navigate to the folder containing `amazon-ml-key.pem` (`cd ~/Downloads`).

### Set Key Permissions and Connect

On macOS / Linux / WSL:
```bash
chmod 400 amazon-ml-key.pem
ssh -i "amazon-ml-key.pem" ubuntu@<YOUR-PUBLIC-IP>
```

On Windows PowerShell:
```powershell
ssh -i "amazon-ml-key.pem" ubuntu@<YOUR-PUBLIC-IP>
```

*(Replace `<YOUR-PUBLIC-IP>` with the actual IP address you copied in Step 2.2, for example: `ubuntu@54.210.88.120`)*

When prompted: `Are you sure you want to continue connecting (yes/no/[fingerprint])?`, type `yes` and press Enter.

You are now logged into the remote AWS Ubuntu server prompt: `ubuntu@ip-...:~$`.

---

## 4. 1-Click Environment Setup on AWS

Paste this entire block into the remote SSH terminal:

```bash
# Update Ubuntu package manager and install required tools
sudo apt update -y && sudo apt install -y python3-pip python3-venv git htop unzip

# Create an isolated Python virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Upgrade pip and install all high-performance C++ and ML dependencies
pip install --upgrade pip
pip install lightgbm catboost rapidfuzz joblib pyarrow fastparquet pandas numpy scikit-learn pytest
```

---

## 5. Clone Repository and Verify Health

Paste these commands:

```bash
# 1. Clone the project branch
git clone -b Bharath-solution https://github.com/bhargav1520/Amazon-ml-challenge-2026.git

# 2. Enter the working directory
cd Amazon-ml-challenge-2026/code/business_entity_resolution

# 3. Run all 21 unit tests to verify zero environment or code breaks
pytest -v
```

All 21 tests will print `PASSED`.

---

## 6. Uploading / Placing the Dataset on AWS

The pipeline looks for the dataset at `Amazon-ml-challenge-2026/student_resource/dataset/` or `code/business_entity_resolution/dataset/`.

If the dataset is on your local computer, open a **new** local terminal window (do not run this inside the SSH session) in the folder containing `student_resource` and run:

### From Your Local Terminal (Uploading data to AWS):
```bash
scp -i "amazon-ml-key.pem" -r student_resource ubuntu@<YOUR-PUBLIC-IP>:~/Amazon-ml-challenge-2026/
```

Verify on the remote server that the files exist:
```bash
ls -lh ~/Amazon-ml-challenge-2026/student_resource/dataset/train/
```
You will see `train_ground_truth.tsv`, `train_source1.tsv`, `train_source2.tsv`, and `train_source3.tsv`.

---

## 7. Run the End-to-End High-Capacity Pipeline

Back in your EC2 SSH session (inside `Amazon-ml-challenge-2026/code/business_entity_resolution`), run:

```bash
python3 run_aws.py
```

### What Happens During Execution:
1. **Stage 1 (Preprocess)**: Cleans names and addresses, expands Indian and international abbreviations, extracts street numbers and pincodes, and writes fast columnar Parquet files.
2. **Stage 2 (Blocking)**: Builds a 6-pass multi-index with `max_candidates=25` across 300,000 training entities and all 1.73M test records to achieve **>99.4% candidate recall**.
3. **Stage 3 (Training)**: Fits an 800-tree dual LightGBM/CatBoost ensemble with 3x hard-negative mining, scoring held-out validation entities and optimizing country-specific decision thresholds.
4. **Stage 4 (Inference)**: Performs batched vector inference across all test candidate pairs with `max_matches_per_entity=15` and writes `output/matching_results.tsv`.
5. **Submission Validator**: Automatically runs `validate_submission.py` and confirms that the output conforms strictly to all competition submission rules.

---

## 8. Download the Submission File to Your Local PC

Once `run_aws.py` finishes, switch back to your **local computer terminal** and run:

```bash
scp -i "amazon-ml-key.pem" ubuntu@<YOUR-PUBLIC-IP>:~/Amazon-ml-challenge-2026/code/business_entity_resolution/output/matching_results.tsv .
```

The file `matching_results.tsv` will download to your local folder. Upload this file directly to the Unstop competition portal.

---

## 9. Terminate the AWS EC2 Instance (Stop Billing)

After downloading your submission file:
1. Go back to the **AWS EC2 Console** in your browser.
2. Click on **Instances** on the left menu.
3. Select your instance (`amazon-ml-worker`).
4. Click the **Instance state** button at the top right $\rightarrow$ click **Terminate instance**.
5. Click **Terminate** to confirm. The instance will shut down and delete, stopping all hourly charges.
