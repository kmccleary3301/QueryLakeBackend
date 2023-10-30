from hashlib import sha256
import random

def random_hash(**kwargs):
    """Returns a random SHA256 hash."""
    return sha256(str(random.getrandbits(512)).encode('utf-8')).hexdigest()

def hash_function(input : str, salt : str = None, only_salt : bool = False) -> str:
    """
    SHA256 hashing function.
    Optional salting. The process is hash(hash(input)+hash(salt)).
    If only_salt is enabled, process is hash(input+hash(salt))
    If no salt is given, process is hash(input)
    """
    if only_salt:
        term_1 = input
    else:
        term_1 = sha256(input.encode('utf-8')).hexdigest()
    if not salt is None:
        salt_term = sha256(salt.encode('utf-8')).hexdigest()
        term_2 = sha256((term_1+salt_term).encode('utf-8')).hexdigest()
        return term_2
    return term_1