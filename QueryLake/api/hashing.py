from hashlib import sha256
import random

HASH_BASE_VOCAB = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def random_hash(length : int = 64, base : int = 16):
    """Returns a random SHA256 hash."""
    assert base >= 2 and base <= len(HASH_BASE_VOCAB), f"Base must be between 2 and {len(HASH_BASE_VOCAB)}."
    
    vocab = HASH_BASE_VOCAB[:base]
    return ''.join(random.choices(vocab, k=length))

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