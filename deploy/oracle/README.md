## Oracle Cloud Always Free (no Git) deployment

This folder contains copy/paste scripts and config templates to deploy this Django app on an Oracle Cloud Always Free Ubuntu VM using:

- Gunicorn (systemd service)
- Nginx (reverse proxy + static files)
- Environment variables in `/etc/movie-reco.env`

### 1) Create the VM (manual in OCI Console)
- Image: **Ubuntu 22.04**
- Shape: **VM.Standard.A1.Flex** (Always Free eligible)
- Networking: ensure the VM has a **public IPv4**
- Ingress rules: allow inbound **TCP 22** (SSH) and **TCP 80** (HTTP)
- Add your SSH public key

When done, you will need:
- **VM_PUBLIC_IP**
- Path to your **SSH private key** (e.g. `~/.ssh/id_rsa` or `~/.ssh/oci_key`)

### 2) Create ZIP locally
From the project root:

```bash
./deploy/oracle/make_zip.sh
```

This creates `movie-reco.zip` in the project root.

### 3) Upload ZIP to VM (no git)

```bash
./deploy/oracle/upload_zip.sh <VM_PUBLIC_IP> ~/.ssh/<your_key>
```

### 4) SSH into the VM and run install
On your Mac:

```bash
ssh -i ~/.ssh/<your_key> ubuntu@<VM_PUBLIC_IP>
```

On the VM:

```bash
cd ~/Movie-Recommendation-System-master
sudo bash ./deploy/oracle/vm_install.sh
```

### 5) Configure secrets (required)
Edit `/etc/movie-reco.env` on the VM:

```bash
sudo nano /etc/movie-reco.env
```

You must set at minimum:
- `SECRET_KEY`
- `ALLOWED_HOSTS=<VM_PUBLIC_IP>`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `TMDB_API_KEY`
- `TMDB_READ_ACCESS_TOKEN`
- `FIREBASE_*` values

### 6) Enable systemd + nginx
On the VM:

```bash
sudo bash ./deploy/oracle/vm_enable_services.sh
```

### 7) Verify
Open in browser:
- `http://<VM_PUBLIC_IP>/`
- `http://<VM_PUBLIC_IP>/api/trending/`
- `http://<VM_PUBLIC_IP>/watchlist/`

