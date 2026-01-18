# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp


def is_diagonally_dominant(A: np.ndarray | sp.sparse.csc_array) -> bool | None:
    """Funkcja sprawdzająca czy podana macierz jest diagonalnie zdominowana.

    Args:
        A (np.ndarray | sp.sparse.csc_array): Macierz A (m,m) podlegająca 
            weryfikacji.
    
    Returns:
        (bool): `True`, jeśli macierz jest diagonalnie zdominowana, 
            w przeciwnym wypadku `False`.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """

    try:
        if not hasattr(A, "shape") or len(A.shape) != 2:
            raise AttributeError

        m, n = A.shape
        if m != n:
            raise AttributeError

        if isinstance(A, sp.sparse.csc_array):
            A_copy: np.ndarray = A.toarray()
        else:
            A_copy: np.ndarray = A.copy()
        
        A_diag: np.ndarray = np.diag(A_copy)

        for i in range(m):
            if 2 * np.abs(A_diag[i]) <= np.sum(np.abs(A_copy[i])):
                return False

        return True
    except:
        return None


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    """Funkcja obliczająca normę residuum dla równania postaci: 
    Ax = b.

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej 
            stronie równania.
    
    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """

    try:
        if len(A.shape) != 2:
            raise AttributeError
    
        m = A.shape[0]
        n = A.shape[1]
        if len(x.shape) != 1 or x.shape[0] != n:
            raise AttributeError
        
        if len(b.shape) != 1 or b.shape[0] != m:
            raise AttributeError
    
        r = b - A @ x
        return np.linalg.norm(r)
    
    except:
        return None
