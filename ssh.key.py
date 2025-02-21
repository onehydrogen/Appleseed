import os
import sys
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import base64

def generate_ssh_key(key_path, key_comment="dashboard-access"):
    """Generate a secure SSH key pair."""
    try:
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )

        # Generate public key
        public_key = private_key.public_key()

        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        # Serialize public key in OpenSSH format
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH
        )

        # Add comment to public key
        public_ssh = public_pem.decode('utf-8') + f" {key_comment}"

        # Save private key
        with open(f"{key_path}/id_rsa", "wb") as f:
            f.write(private_pem)
        os.chmod(f"{key_path}/id_rsa", 0o600)

        # Save public key
        with open(f"{key_path}/id_rsa.pub", "w") as f:
            f.write(public_ssh)
        os.chmod(f"{key_path}/id_rsa.pub", 0o644)

        return True, "SSH key pair generated successfully."

    except Exception as e:
        return False, f"Error generating SSH key: {str(e)}"

# Set up SSH key directory
home_dir = os.path.expanduser("~")
ssh_dir = os.path.join(home_dir, ".ssh")

# Create .ssh directory if it doesn't exist
if not os.path.exists(ssh_dir):
    os.makedirs(ssh_dir, mode=0o700)

# Generate the key pair
success, message = generate_ssh_key(ssh_dir, "legislative-dashboard")
print(message)

if success:
    print(f"\nSSH key pair has been generated:")
    print(f"Private key: {ssh_dir}/id_rsa")
    print(f"Public key: {ssh_dir}/id_rsa.pub")
    print("\nYour public key content is:")
    with open(f"{ssh_dir}/id_rsa.pub", "r") as f:
        print(f.read())