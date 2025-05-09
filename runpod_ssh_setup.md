
# ✅ Manual SSH Setup for RunPod (Exposed TCP)

> Use this guide if `scp`/`ssh` isn't working out of the box.

---

## 🔹 Step 1: Generate SSH key (if you don't have one)

Run this **on your local machine**:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press `Enter` to accept defaults. This creates:
- `~/.ssh/id_ed25519` (private key)
- `~/.ssh/id_ed25519.pub` (public key)

---

## 🔹 Step 2: Add your public key to RunPod

1. Open your pod in [RunPod Console](https://runpod.io/console)
2. Scroll to **“SSH over exposed TCP”**
3. Paste the output of:

```bash
cat ~/.ssh/id_ed25519.pub
```

4. Click **Save**

---

## 🔹 Step 3: SSH into the pod via browser terminal

If your key still isn't accepted yet, SSH into the pod using **RunPod's web terminal**.

Then run:

```bash
mkdir -p /root/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFjk+49FF52qTv2mYUS4m3IVMSIsKG6nA2ifholdQ9xu uday.rub3@gmail.com" >> /root/.ssh/authorized_keys
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
```

> Replace `ssh-ed25519 AAAA...` with your actual public key (on one line)

---

## 🔹 Step 4: Ensure SSH service is running

In the terminal, run:

```bash
ps aux | grep sshd
```

If it's not running, start it:

```bash
apt update && apt install openssh-server -y
service ssh restart
```

---

## 🔹 Step 5: Connect from local machine

```bash
ssh -p <port> -i ~/.ssh/id_ed25519 root@<runpod-ip>
```

Example:

```bash
ssh -p 22058 -i ~/.ssh/id_ed25519 root@194.68.245.65
```

---

## 🔹 Step 6: Upload file with `scp`

From your **local machine**, run:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 yourfile.zip root@<runpod-ip>:/workspace
```

Example:

```bash
scp -P 22058 -i ~/.ssh/id_ed25519 battery-rag-llm.zip root@194.68.245.65:/workspace
pip install -U sentence-transformers transformers accelerate bitsandbytes ninja datasets peft trl tabulate tqdm langchain_community ragas
huggingface-cli login
```
