from ecies.utils import generate_eth_key, generate_key
from ecies import encrypt, decrypt
# from .authentication import hash_function, get_random_hash
from Crypto.Cipher import AES
from hashlib import sha256
import random
from fastapi import UploadFile
import py7zr
import os
from io import BytesIO
from typing import Dict, Union


def get_random_hash():
    return sha256(str(random.getrandbits(512)).encode('utf-8')).hexdigest()

def hash_function(input : str, salt : str = None, only_salt : bool = False) -> str:
    if only_salt:
        term_1 = input
    else:
        term_1 = sha256(input.encode('utf-8')).hexdigest()
    if not salt is None:
        salt_term = sha256(salt.encode('utf-8')).hexdigest()
        term_2 = sha256((term_1+salt_term).encode('utf-8')).hexdigest()
        return term_2
    return term_1

def ecc_generate_public_private_key() -> tuple[str]:
    """
    Generates a random public-private key pair, both as hex strings.
    Returns as (public_key, private_key)
    """

    eth_k = generate_eth_key()
    sk_hex = eth_k.to_hex()  # hex string
    pk_hex = eth_k.public_key.to_hex()

    return (pk_hex, sk_hex)

def ecc_encrypt_string(public_key_hex : str, input_string : str, encoding : str = "utf-8") -> str:
    """
    Encrypts input string using a public key hex string, as generated by ecc_generate_public_private_key().
    Data is returned as a hex string.
    """
    data = bytes(input_string, encoding=encoding)
    data_encrypted = encrypt(public_key_hex, data).hex()
    return data_encrypted

def ecc_decrypt_string(private_key_hex : str, encrypted_hex_string : str, encoding : str = "utf-8") -> str:
    """
    Decrypts encrypted hex string using a private key hex string, as generated by ecc_generate_public_private_key().
    """
    data = decrypt(private_key_hex, bytes.fromhex(encrypted_hex_string)).decode(encoding=encoding)
    return data

def aes_encrypt_string(key : str, input_string : str, encoding : str = "utf-8") -> str:
    """
    Encrypts input string using any input key string.
    Data is returned as a hex string.
    """
    # key = bytes.fromhex(hash_function(key))
    key = bytes("abcdef0123456789", encoding="utf-8")
    nonce = get_random_hash()
    obj = AES.new(key, AES.MODE_EAX, bytes.fromhex(nonce))
    ciphertext = obj.encrypt(bytes(input_string, encoding=encoding))
    return ciphertext.hex()+nonce

def aes_decrypt_string(key : str, encrypted_hex_string : str, encoding : str = "utf-8") -> str:
    """
    Decrypts ecnrypted hex string using any input key string.
    """
    key = bytes("abcdef0123456789", encoding="utf-8")
    nonce = encrypted_hex_string[-64:]
    encrypted_hex_string = encrypted_hex_string[:-64]
    obj = AES.new(key, AES.MODE_EAX, bytes.fromhex(nonce))
    ciphertext = obj.decrypt(bytes.fromhex(encrypted_hex_string))
    return ciphertext.decode(encoding=encoding)

def zip_test(key : str):

    current_directory = os.getcwd()


    save_path = "/home/user/python_projects/3035/target_extract"
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])+"/"
    with py7zr.SevenZipFile('target.7z', mode='r', password=key) as z:
        # os.chdir(save_path)
        z.extractall("target_di")
    os.chdir(current_directory)

    with py7zr.SevenZipFile('target.7z', 'w', password=key, header_encryption=True) as z:
        z.write(server_dir+"QueryLake")

def aes_encrypt_zip_file(key : str, file_data : Dict[str, Union[str, bytes]], save_path : str):
     with py7zr.SevenZipFile(save_path, 'w', password=key, header_encryption=True) as z:
        z.writed(file_data)

def aes_decrypt_zip_file(key : str, file_path : str):
    """
    Returns dictionary with structure of archive.
    Each file value is a BytesIO object.
    """

    with py7zr.SevenZipFile(file_path, mode='r', password=key) as z:
        return z.read()

# def save_file_aes(file_path : str, encryption_key : str) -> None:



if __name__ == "__main__":
    test_key = "dkljafsldkjflasjhfie8rwher82u4982u39"
    test_msg = "get_randomakljdlfkasjkdjfhLorem ipsum马云Lorem ipsum马云Lorem ipsum马云Lorem ipsum马云Lorem ipsum马云"
    # encrypted_string = aes_encrypt_string(test_key, test_msg)
    # decrypted_string = aes_decrypt_string(test_key, encrypted_string)
    # print(decrypted_string)
    aes_encrypt_zip_file(test_key, {"Test.txt": BytesIO(bytes(test_msg, encoding="utf-8"))}, "test_create_encrypted_file.7z")
    print(aes_decrypt_zip_file(test_key, "test_create_encrypted_file.7z"))
    # zip_test(test_key)