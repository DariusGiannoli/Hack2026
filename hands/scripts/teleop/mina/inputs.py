"""
Mock minimal du module 'inputs' pour les environnements sans manette.
Se2Gamepad appelle get_gamepad() dans son thread de fond — on retourne
une liste vide pour que la boucle ne plante pas.
"""


def get_gamepad():
    """Retourne une liste vide (aucun événement manette)."""
    import time
    time.sleep(0.1)   # évite le busy-loop si appelé en continu
    return []
