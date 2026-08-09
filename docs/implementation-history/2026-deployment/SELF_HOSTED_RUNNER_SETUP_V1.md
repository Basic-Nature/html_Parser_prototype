# Self-Hosted GitHub Actions Runner Setup for BallotLens VNet

**Objective**: Provision a dedicated Linux VM in your Azure VNet to act as a self-hosted GitHub Actions runner for warehouse seeding operations.

**Timeline**: ~30 minutes total
**Cost Impact**: ~$8–15/month for Standard_B2s VM (adjust based on region)

---

## Prerequisites Checklist

- [ ] GitHub token with `admin:org_self_hosted_runner` scope (create at <https://github.com/settings/tokens?type=beta>)
- [ ] Access to Azure Portal or Terraform for your subscription
- [ ] Knowledge of your VNet name and subnet name where PostgreSQL resides
- [ ] Administrator access to create/modify NSGs
- [ ] SSH client on your local machine (or Azure Bastion for break-glass access)

---

## Phase 1: Create the Linux VM in Azure VNet

### Step 1.1: Gather VNet Information

Open Azure Portal and locate your existing PostgreSQL Flexible Server to confirm VNet topology:

1. Navigate to **Azure Portal** → search "ballotlens-server" (PostgreSQL resource)
2. Click on the resource
3. Note the **Networking** blade:
   - **Virtual network**: `vnet-wwiyjoju`
   - **Subnet**: `subnet-3dufemsymus4k` (delegated to PostgreSQL only)
   - **Private DNS zone**: `privatelink.postgres.database.azure.com`

### Important Constraint

1. `subnet-3dufemsymus4k` is delegated to `Microsoft.DBforPostgreSQL/flexibleServers`.
2. You **cannot** place the runner VM NIC in this delegated subnet.
3. Create a **separate runner subnet** in the same VNet (for example `runner-subnet`) and place the VM there.
4. Keep PostgreSQL access by FQDN (`ballotlens-server.postgres.database.azure.com`) through Private DNS.

**Record these values for Step 1.3:**

```txt
VNet Name: vnet-wwiyjoju
PostgreSQL Subnet (delegated): subnet-3dufemsymus4k
Runner Subnet (create new): ___________________
Private DNS Zone: privatelink.postgres.database.azure.com
Resource Group: BallotLens_group
```

### Step 1.2: Create Network Security Group (NSG) for Runner VM

The NSG will restrict inbound/outbound traffic to only what the runner needs.

Before creating the VM, create a **new subnet for the runner** in the same VNet:

1. Go to **Virtual networks** → `vnet-wwiyjoju` → **Subnets** → **+ Subnet**
2. Name: `runner-subnet`
3. Address range: pick a free CIDR block in your VNet (for example `10.0.2.0/24`)
4. **Delegation**: `None`
5. Save

**Via Azure Portal:**

1. Search **Network security groups** in Azure Portal
2. Click **+ Create**
3. Fill in:
   - **Name**: `runner-ingestion-nsg`
   - **Subscription**: (your subscription)
   - **Resource Group**: `BallotLens_group` (same as PostgreSQL)
   - **Region**: (same as your VNet, typically `eastus` or your region)
4. Click **Review + Create** → **Create**

**After creation, add inbound rules:**

Open the newly created NSG → **Inbound security rules** → click **+ Add**

_**Rule 1: Allow SSH from Azure Bastion (or admin IP)**_

If using **Azure Bastion** (recommended):

- **Source**: Service Tag → `AzureBastion`
- **Source port ranges**: `*`
- **Destination**: `*`
- **Destination port ranges**: `22`
- **Protocol**: TCP
- **Action**: Allow
- **Priority**: `100`
- **Name**: `AllowSSHFromBastion`

If using **Static Admin IP** instead:

- **Source**: IP Addresses → `YOUR_ADMIN_IP/32`
- **Source port ranges**: `*`
- **Destination**: `*`
- **Destination port ranges**: `22`
- **Protocol**: TCP
- **Action**: Allow
- **Priority**: `100`
- **Name**: `AllowSSHFromAdminIP`

_**Rule 2: Allow packet flow within VNet (inter-VM communication)**_

- **Source**: Virtual Network
- **Source port ranges**: `*`
- **Destination**: Virtual Network
- **Destination port ranges**: `*`
- **Protocol**: Any
- **Action**: Allow
- **Priority**: `101`
- **Name**: `AllowVNetInternal`

**Outbound rules** (should already allow by default, but verify):

Open **Outbound security rules** and confirm:

- A default **Allow** rule exists for destination `*` on ports `*`
- If not, add:
  - **Destination**: Internet (or specific IPs below)
  - **Destination port ranges**: `443, 80` (HTTPS/HTTP for GitHub, PyPI, Google APIs)
  - **Protocol**: TCP
  - **Action**: Allow
  - **Priority**: `100`
  - **Name**: `AllowOutboundHTTPS`

**Specific outbound allowlist** (optional, more restrictive):

If your org requires strict outbound filtering, add these IPs/domains to firewall rules (contact your network team):

- GitHub API: `api.github.com` (130.199.5.0/24, plus others — check <https://docs.github.com/en/authentication-and-security/keeping-your-account-and-data-secure/about-github-s-ip-addresses>)
- PyPI: `pypi.org` (18.218.232.0/22, etc.)
- Google APIs: `googleapis.com` (various ranges)
- PostgreSQL private endpoint: `ballotlens-server.postgres.database.azure.com:5432` (should be internal VNet routing, not internet)

For now, **allow outbound 443 (HTTPS)** to all destinations to unblock; tighten later if needed.

### Step 1.3: Create the Runner VM

**Via Azure Portal:**

1. Search **Virtual machines** → **+ Create** → **Azure virtual machine**
2. **Basics** tab:
   - **Subscription**: (your subscription)
   - **Resource Group**: `BallotLens_group`
   - **Virtual machine name**: `runner-ingestion-01`
   - **Region**: (same as your VNet, e.g., `eastus`)
   - **Availability options**: No infrastructure redundancy needed
   - **Image**: Ubuntu Server 22.04 LTS - x64 Gen2
   - **VM architecture**: x64
   - **Size**: Standard_B2s (2 vCPUs, 4 GB RAM)
   - **Authentication type**: SSH public key (recommended) or password
     - If SSH key: upload your public key or generate new
     - If password: set a strong password
3. Click **Next: Disks**
   - **OS disk type**: Standard SSD (sufficient for light runner workloads)
   - Click **Next: Networking**

4. **Networking** tab:
   - **Virtual network**: `vnet-wwiyjoju`
   - **Subnet**: `runner-subnet` (or your non-delegated runner subnet from Step 1.2)
   - **Public IP**: **None** (keep runner private; use Bastion for SSH)
   - **NIC network security group**: **Custom** → select `runner-ingestion-nsg` (created in Step 1.2)
5. Click **Next: Management**
   - **Enable guest OS diagnostics**: No (optional for cost savings)
   - Click **Review + Create**

6. Review summary and click **Create**

**Wait for deployment** (~2–3 minutes). Once complete, note the **Private IP** of the runner VM:

- Open the VM resource → **Networking** blade
- Copy the **Private IP address** (typically 10.x.x.x)

```txt
Runner VM Private IP: ___________________
```

### Step 1.4: Verify Private Network Connectivity (Optional but Recommended)

Once the VM is created, you can quickly verify it can reach PostgreSQL:

**From your local machine via Azure Bastion** (if Bastion is set up):

1. Open Azure Portal → search "runner-ingestion-01" → click **Connect** → **Bastion**
2. Authenticate and open terminal
3. Run:

   ```bash
    nslookup ballotlens-server.postgres.database.azure.com
    nc -zv ballotlens-server.postgres.database.azure.com 5432
    ```

    - `nslookup` should resolve through the private DNS zone (`privatelink.postgres.database.azure.com`) to a private RFC1918 IP.
    - `nc` should show: `Connection to ballotlens-server.postgres.database.azure.com 5432 port [tcp/postgresql] succeeded!`

If `nslookup` returns a public IP, DNS zone linking is wrong. Verify:

1. `privatelink.postgres.database.azure.com` is linked to `vnet-wwiyjoju`.
2. Runner VM DNS uses Azure-provided resolver (or custom DNS that forwards Azure private zones).

If you don't have Bastion or this fails, that's OK — we'll verify connectivity during runner registration.

### Step 1.5: Find Private IPs the right way

Use this only for diagnostics; the workflow should use FQDN, not a hardcoded DB IP.

- Runner VM private IP:

   ```bash
   az vm list-ip-addresses -g BallotLens_group -n runner-ingestion-01 --query "[0].virtualMachine.network.privateIpAddresses[0]" -o tsv
   ```

- PostgreSQL effective private IP (DNS-resolved):

   ```bash
   nc -zv ballotlens-server.postgres.database.azure.com 5432
   nslookup ballotlens-server.postgres.database.azure.com
   ```

   The DB IP may change; rely on the server name and Private DNS for durable connectivity.

---

## Phase 2: Install GitHub Actions Runner on the VM

### Step 2.1: SSH into the Runner VM

Use Azure Bastion or your SSH key:

**Via Bastion (easiest):**

1. Azure Portal → search "runner-ingestion-01"
2. Click **Connect** → **Bastion**
3. Username: `azureuser` (default for Ubuntu)
4. Authentication: SSH private key (if key-based) or password

**Via SSH from local machine** (requires public IP or port forwarding):

```bash
ssh -i /path/to/private/key azureuser@<runner_vm_public_ip>
```

(If no public IP, use Bastion method above.)

### Step 2.2: Prepare the VM Environment

Once connected, run:

```bash
# Update package manager
sudo apt-get update
sudo apt-get upgrade -y

# Install dependencies for GitHub runner and Python
sudo apt-get install -y \
  git \
  curl \
  wget \
  jq \
  build-essential \
  libssl-dev \
  libffi-dev \
  python3-dev \
  python3-pip \
  postgresql-client \
  net-tools

# Create runner directory
mkdir -p /opt/actions-runner
cd /opt/actions-runner

# Verify Python 3.12 (or install if not present)
python3 --version
# If older than 3.12, optionally upgrade (beyond scope here; Ubuntu 22.04 LTS has 3.10, which is fine for this role)
```

### Step 2.3: Download and Extract GitHub Actions Runner

1. Check the latest runner release at <https://github.com/actions/runner/releases>
2. Find the URL for `actions-runner-linux-x64-*.tar.gz` (latest stable)
3. Download (example version 2.323.0; **use actual latest**):

```bash
cd /opt/actions-runner

# Download (replace version number if newer is available)
curl -o actions-runner-linux-x64-2.323.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.323.0/actions-runner-linux-x64-2.323.0.tar.gz

# Extract
tar xzf actions-runner-linux-x64-2.323.0.tar.gz

# Verify
ls -la
# Should see: bin/, externals/, run.sh, config.sh, etc.
```

### Step 2.4: Generate GitHub Personal Access Token (PAT)

You need a PAT to register the runner. Create one at:

1. Go to <https://github.com/settings/tokens?type=beta>
2. Click **Generate new token** (beta)
3. Set:
   - **Token name**: `runner-registration-token-2026`
   - **Expiration**: 90 days (rotate before expiration)
   - **Repository access**: Select repository → `Basic-Nature/html_Parser_prototype`
   - **Permissions**:
     - Under **Administration**: check ✓ `self-hosted runners`
4. Copy the token and paste it into a **secure location** (e.g., `~/.github_runner_token` on your local machine — **NEVER commit to git**)

```txt
GitHub PAT: ___________________
```

### Step 2.5: Configure the Runner

Back on the runner VM, run the config script:

```bash
cd /opt/actions-runner

# Run configuration (replace <PAT> and <ADMIN_IP> below)
./config.sh \
  --url https://github.com/Basic-Nature/html_Parser_prototype \
  --token <YOUR_PAT_FROM_STEP_2.4> \
  --name runner-ingestion-01 \
  --labels self-hosted,linux,x64,ballotlens-vnet \
  --work _work \
  --replace \
  --unattended

# Expected output:
# -------------------- Actions Runner Configuration --------------------
# |           ACTIONS_RUNNER_INPUT_URL  https://github.com/Basic-Nature/html_Parser_prototype
# |         ACTIONS_RUNNER_INPUT_TOKEN  ***
# |        ACTIONS_RUNNER_INPUT_RUNNERGROUP
# ...
# Runner successfully configured for use with GitHub Actions
```

**Breakdown of flags:**

- `--url`: Your repository URL
- `--token`: GitHub PAT from Step 2.4
- `--name`: Friendly name for the runner (appears in repo settings)
- `--labels`: Custom labels (critical: must match `runs-on:` in workflow)
  - `self-hosted`: marks it as self-hosted
  - `linux`, `x64`: OS markers
  - `ballotlens-vnet`: custom label for your VNet role
- `--replace`: Overwrite if runner already exists at this path
- `--unattended`: Don't prompt for input (needed for systemd service)

### Step 2.6: Install Runner as Systemd Service

Run:

```bash
cd /opt/actions-runner

# Install service (requires sudo)
sudo ./svc.sh install

# Start the service
sudo ./svc.sh start

# Verify it's running
sudo systemctl status actions.runner.Basic-Nature-html_Parser_prototype.runner-ingestion-01.service

# Expected output:
# ● actions.runner.Basic-Nature-html_Parser_prototype.runner-ingestion-01.service - GitHub Actions Runner
#    Loaded: loaded (.../etc/systemd/system/actions.runner.*.service; enabled; vendor preset: enabled)
#    Active: active (running)
```

**Service will auto-start on VM reboot** (due to `--unattended` flag in Step 2.5).

### Step 2.7: Enable Auto-Update (Optional but Recommended)

The runner can auto-update itself. Enable it:

```bash
cd /opt/actions-runner

# Enable auto-update (won't prompt user)
./config.sh --url https://github.com/Basic-Nature/html_Parser_prototype \
  --token <YOUR_PAT_FROM_STEP_2.4> \
  --replace \
  --unattended \
  --runnergroup default
```

Alternatively, if already configured, the runner will automatically update during idle periods.

---

## Phase 3: Verify Runner Registration

### Step 3.1: Check GitHub Repo Settings

1. Go to <https://github.com/Basic-Nature/html_Parser_prototype>
2. Navigate to **Settings** → **Actions** → **Runners**
3. Look for `runner-ingestion-01` in the list
4. **Confirm**:
   - Status: **Idle** (green circle)
   - Labels: `self-hosted`, `linux`, `x64`, `ballotlens-vnet`
   - Last activity: "Just now" or recent timestamp

### Step 3.2: Test Runner Connectivity (Optional)

Create a simple test workflow to confirm the runner can execute:

1. Create `.github/workflows/test-runner.yml`:

   ```yaml
   name: Test Runner
   on: workflow_dispatch
   jobs:
     test:
       runs-on: [self-hosted, linux, x64, ballotlens-vnet]
       steps:
         - run: echo "Runner is online and executing!"
         - run: hostname
         - run: uname -a
   ```

2. Push to main branch
3. Go to **Actions** tab → find **Test Runner** → click **Run workflow**
4. Should execute immediately on your runner VM (not wait for GitHub runners)

---

## Phase 4: Verify PostgreSQL Private Connectivity from Runner

### Step 4.1: Quick TCP Port Check

SSH back into the runner VM:

```bash
sudo nc -zv ballotlens-server.postgres.database.azure.com 5432
# Should show: Connection ... succeeded!
```

If this fails, your NSG or VNet routing is misconfigured. Check:

- NSG outbound rules allow 5432
- Runner VM is in `runner-subnet` (non-delegated) within `vnet-wwiyjoju`
- Private DNS zone link exists: `privatelink.postgres.database.azure.com` → `vnet-wwiyjoju`
- Runner VM subnet routing tables include the PostgreSQL delegated subnet

### Step 4.2: Test a Real PostgreSQL Query (Optional)

```bash
# Install psql if not already installed
sudo apt-get install -y postgresql-client

# Test connection to PostgreSQL
psql -h ballotlens-server.postgres.database.azure.com \
     -U <POSTGRES_USER> \
     -d ballotlens-database \
     -c "SELECT COUNT(*) FROM warehouse_election_results LIMIT 1;"

# Prompted for password — enter <POSTGRES_PASSWORD> from your GitHub Secrets
```

If this succeeds, your runner can fully access the warehouse database.

---

## Phase 5: Clean-Up & Security Hardening

### Step 5.1: Rotate and Secure the PAT

1. The PAT from Step 2.4 is now **embedded in `/opt/actions-runner/.credentials`** on the runner VM.
2. Do **NOT** share this token.
3. After confirming the runner works, optionally rotate the token:
   - Go to <https://github.com/settings/tokens>
   - Delete the old runner token
   - Create a new one if needed for future runner registration

### Step 5.2: Lock Down VM SSH Access

Confirm SSH is restricted to Bastion or admin IP:

```bash
# Verify NSG rule
sudo ufw status  # If UFW is in use
# Or check Azure Portal NSG inbound rules
```

### Step 5.3: Document Runner Credentials

**Create a secure runbook** (NOT in Git):

```txt
Runner VM Credentials & Details
- VM Name: runner-ingestion-01
- Private IP: <from Step 1.3>
- Resource Group: BallotLens_group
- SSH User: azureuser
- SSH Key Location: (your key path)
- Runner Labels: self-hosted, linux, x64, ballotlens-vnet
- PostgreSQL Test: psql -h ballotlens-server.postgres.database.azure.com -U <user> -d ballotlens-database
- Restart Runner Service: sudo systemctl restart actions.runner.Basic-Nature-html_Parser_prototype.runner-ingestion-01.service
```

---

## Troubleshooting

| Issue | Cause | Fix |
| ------- | ------- | ----- |
| Runner shows "Offline" in GitHub | Service stopped | SSH to VM: `sudo systemctl start actions.runner.Basic-Nature-html_Parser_prototype.runner-ingestion-01.service` |
| Workflow job hangs waiting for runner | Runner not registered or wrong labels | Verify runner appears in repo Settings → Runners; check label names match `runs-on:` |
| VM creation fails in DB subnet | Subnet is delegated to PostgreSQL Flexible Server | Create `runner-subnet` with no delegation and deploy runner VM there |
| PostgreSQL connection timeout in verification step | NSG blocks 5432, wrong subnet, or DNS link missing | Verify NSG outbound allows 5432, VM is in non-delegated subnet in `vnet-wwiyjoju`, and Private DNS zone link exists |
| `nc -zv ballotlens-server` fails | Private DNS not resolving or route issue | Run `nslookup ballotlens-server.postgres.database.azure.com`; if not private IP, fix Private DNS zone link/forwarding |
| Runner service fails to start | Config script failed or permissions issue | Check logs: `sudo journalctl -u actions.runner.Basic-Nature-html_Parser_prototype.runner-ingestion-01.service -n 50` |

---

## Post-Runner Setup: Update Workflow for Self-Hosted

Once the runner is confirmed **Idle** in GitHub Settings, the seed workflow will automatically pick it up:

**Workflow already configured for this runner:**

- File: `.github/workflows/seed-warehouse.yml`
- `runs-on: [self-hosted, linux, x64, ballotlens-vnet]`
- First run will pull imports from PyPI and compile; subsequent runs cache dependencies.

**To test the workflow with your new runner:**

1. Go to **Actions** tab
2. Select **Seed Warehouse (manual data import)**
3. Click **Run workflow** button
4. Fill in inputs:
   - `import_finalized_data`: `true`
   - `import_voting_equipment`: `false`
   - `dry_run`: `false`
   - `worksheet`: `Finalized Data`
   - `verify_post_seed`: `true`
   - `verify_timeout_sec`: `20`
   - `min_expected_warehouse_rows`: `200000`
   - `min_expected_distinct_states`: `45`
5. Click **Run workflow**
6. Monitor in **Actions** → **Seed Warehouse** active job

---

## Estimated Timeline

| Phase | Task | Time |
| ------- | ------- | ----- |
| 1 | Create NSG, gather VNet info | 5 min |
| 1 | Create VM | 3 min (wait) |
| 2 | SSH to VM, install deps | 2 min |
| 2 | Download and configure runner | 3 min |
| 2 | Install as systemd service | 1 min |
| 3 | Verify runner in GitHub Settings | 2 min |
| 4 | Test PostgreSQL connectivity | 2 min |
| **Total** | | **~18 min** |

---

## Next Steps

Once the runner is **Idle** in GitHub Settings:

1. **Run the first Seed Warehouse workflow** (with dry_run=false) and share the Post-Seed SQL Verification output
2. **Baseline the verification results** to inform permanent threshold values
3. **Enable margin-based fail mode** after 2–3 stable runs
4. **Rotate PostgreSQL credentials** after confirmed successful seed

---

**Questions?** Refer back to the prerequisites checklist and troubleshooting table, or reach out with specific error messages.
