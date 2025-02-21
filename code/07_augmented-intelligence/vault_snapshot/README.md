### Obsidian Vault Snapshot Tool**  

---

## **📌 Overview**  
This Python module creates **a snapshot copy of your Obsidian vault** while:  
✔ **Only copying Markdown (`.md`) files**  
✔ **Excluding files or folders** based on a Git-ignore style pattern  
✔ **Overwriting the target directory** to keep the snapshot fresh  
✔ **Preventing accidental deletion** of your main vault by ensuring the target is not inside the vault  

**💡 Why?**  
This is useful when experimenting with **AI on your notes** while keeping your original vault **safe**.

---

## **⚙️ Installation**
### **Clone the Repository**
```bash
git clone https://github.com/yourusername/vault_snapshot.git
cd vault_snapshot
```

## **📝 Configuration**
Before running the script, create a `config.yaml` file that defines:  
- The **source vault directory**  
- The **target snapshot location**  
- A list of **ignored files and folders**  

### **Example: `config.yaml`**
```yaml
vault_path: "/Users/chris/ObsidianVault"  # Your main Obsidian vault
target_dir: "/Users/chris/ObsidianVault_Backup"  # Snapshot location
exclude:
  - "private_*.md"  # Ignore any markdown file starting with "private_"
  - "drafts/*"  # Ignore everything inside the drafts folder
```

---

## **📌 Usage**
### **1️⃣ Run from CLI**
#### **Basic Usage (Use Defaults)**

Uses the `config.yaml` in root location

```bash
python vault_snapshot/snapshot 
```

#### **Specify a Config File**
```bash
python -m vault_snapshot.snapshot --config /path/to/config.yaml
```

---

### **2️⃣ Use in Python**
#### **Run the Snapshot from a Script**
```python
from vault_snapshot.snapshot import snapshot

snapshot("/Users/chris/ObsidianVault_Backup")
snapshot() # will use the default config.yaml
```

#### **Run with a Custom Config File**
```python
snapshot(target_dir="/Users/chris/ObsidianVault_Backup", config_path="/path/to/my_config.yaml")
```
---

## **✅ Safety Checks**
🛑 **Prevents accidental deletion of your main vault**  
✔ **Never allows the target directory to be inside the vault**  
✔ **Never allows overwriting the vault itself**  

If you attempt to set the target inside the vault, you’ll see:  
```
ValueError: Target directory cannot be the same as, or inside, the vault directory.
```


---

## **🧪 Running Tests**
This module includes **automated tests** using `pytest`.

### **Run all tests**
```bash
pytest tests/
```

---

## **🛠️ Development**
If you want to modify or improve this tool:  
- Fork or clone the repo  
- Edit `vault_snapshot/snapshot.py`  
- Run `pytest` before submitting changes  

---

## **🔗 License**
This project is **open-source** under the MIT license.

---

This README provides **quick setup, clear usage examples, and safety guarantees**. 🚀 Let me know if you'd like any refinements!