### CALCUL COEFFICIENTS ###

# Import des bibliothèques

import time
import numpy as np
import matplotlib.pyplot as plt

# Ouverture du document texte

f = open("aigle3.txt", 'r')

### Programme Principal

# Convertir en liste

def OuvrEtConv (f) :

    '''

    entrée : str
    sortie : list

    but : prend en entrée la chaîne de caractères du type :
    0.0066
    0.0095
    qui représente les ordonnées de la fonction tracée par le signal et qui renvoie la liste de ces valeurs

    '''
    c = f.read()
    M = []
    i = 0 #itération des caractères
    z = 0 #compteur de valeurs

    for k in range (len(c)) : # on compte ici le nombre de valeurs
        if c[k] == '\n' :
            z += 1

    for j in range (z) : #j est donc l'indice de la valeur

        P = 0 #valeur entière
        D = 0 #valeur décimale
        V = 0 #valeur complète
        a = 0 #indique si la valeur est négative ou non

        if c[i] == '-' : # valeur négative ou non
            a += 1
            i += 1
        N = [] #liste comportant les caractères composant la partie entière
        while c[i] != '.' : #on cherche la partie entière
            N.append(c[i])
            i += 1
        for k in range (len(N)) :
            P += float(N[-k])*(10**(k+1))
        i += 1
        L = [] #même chose avec la partie décimale
        while c[i] != '\n' :
            L.append(c[i])
            i += 1
        for k in range (len(L)) :
            D += int(L[k])*(10**(-k-1))
        if a == 1 :
            V = (-1)*(P + D)
        else :
            V = P + D
        i += 1
        if abs(V) >= 0.0001 : #pour prendre que les valeurs utiles
            M.append(V)

    return(M)

# Valeur moyenne

def VaMoy (L) :

    '''

    entrée : list
    sortie : float

    but : renvoie la valeur moyenne de la liste d'entrée

    '''

    a = 0 #sommme des valeurs
    for x in L:
        a+=x

    return a/len(L)

# Période

def Periode (L) :

    '''

    entrée : list
    sortie : list

    but : renvoie la liste des différentes périodes du signal (liste d'entrée)

    '''

    n = len(L)
    f1 = 0 #début de la période
    f2 = 0 #fin de la période
    P = [] #liste des périodes
    a = 0 #nombre de fois que la valeur moyenne est vue (pour une période)
    VM = VaMoy(L)

    for i in range(n-1) :
        if (L[i] <= VM and L[i+1] > VM) or (L[i] >= VM and L[i+1] < VM) : #si la fonction passe par sa valeur moyenne
            if a == 0 :
                a += 1
                f1 = i
            elif a == 1 :
                a += 1
            elif a == 2 :
                f2 = i - 1
                a = 0
                P.append(f2-f1)

    return P

# Échantillonnage

def Echan (L,points) :

    '''

    entrée : list
    sortie : list

    but : renvoie la liste échantillonnée, avec ici 'points' points par période, de la liste d'entrée

    '''

    # on essaie avec 'points' point par périodes

    T = VaMoy(Periode(L))#fréquence moyenne
    LE = [] #signal échantillonné
    NbT = int(len(L)/T) #nombre de périodes
    if NbT*points >= len(L) :
        print("échantillonnage impossible, renvoi de la liste de départ non échantillonnée")

        return L

    else :
        for i in range (NbT*points) :
            LE.append(L[int(i*(T//points))])

        return LE

# Normaliser

def SiNor (L) :

    '''

    entrée : list
    sortie : list

    but : renvoie la liste d'entrée décalée verticalement, de telle sorte que sa valeur moyenne soit nulle

    '''
    LN = [] #nouvelle liste
    VM = VaMoy(L)
    for i in range (len(L)) :
        LN.append(L[i] - VM)

    return LN

# Pré-accentuer

def SiAcc (L,LN) : #on accentue certaines fréquences

    '''

    entrée : list,list
    sortie : list

    but : renvoie la liste d'entrée (LN) où les hautes fréquences sont accentuées

    '''

    K = 0.95 #facteur d'accentuation

    LA = [VaMoy(L)] #on prend comme première valeur, la valeur moyenne du signal non normalisé
    for i in range (len(L)-1) :
        LA.append(LN[i+1]-K*LN[i])

    return LA

# Trames

def Trame (L,LF) : #avec L la liste de base et LF la liste échantillonnée

    '''

    entrée : list,list
    sortie : list(list)

    but : renvoie la liste des différents échantillions de la liste d'entrée (LF)

    '''

    T = VaMoy(Periode(L)) #période moyenne
    Q = int(220.5/(len(LF)/len(L))) #longueur d'un trame
    q = int(Q*2) #longueur de l'échantillion
    end = int(len(LF)/Q) #nombre total possible de trames
    LT = [] #liste regroupant toutes les trames
    K = 0

    while K < end - 1 : #jusqu'a avoir fait le nombre maximal de trames
        Ltemp = [] #trame
        for i in range (Q*K,Q*K + q) :
            tra = L[i]*FoncFen(K-i,q)
            'note : si problème par la suite, on peut essayer avec (K*Q+q) à la place de K'
            Ltemp.append(tra)
        LT.append(Ltemp)
        K += 1

    return LT

def FoncFen (n,K) :

    '''

    entrée : int,int
    sortie : float

    '''

    return 0.54-0.46*np.cos(2*np.pi*((n-1)/(K-1)))

# Fourier

def Rajoute_0 (L) :

    '''

    entrée : list
    sortie : list

    but : renvoie une liste dont le début est la liste d'entrée et la fin est des zéros, de telle sorte que la taille de la liste soit une puissance de 2

    '''

    x = 0 #test
    i = 0 #itération dans la liste
    n = len(L)

    while x == 0 : # on cherche la puissance de deux supérieure la plus proche
        if n == 2**i :
            x = 1
        elif 2**i < n and 2**(i+1) > n : # si la puissance de deux supérieure la plus proche est 2 puissance i alors :
            x = 1
        i += 1
    r = (2**i)-n #le nombre de 0 à rajouter
    L_new = []
    for k in L :
        L_new.append(k)
    for k in range (r) :
        L_new.append(0)

    return L_new

def FFT(x):

    '''

    entrée : list
    sortie : list

    but : renvoie la transformée de Fourier de x, version numpy

    '''

    # on se base sur la symétrie de la transformée de Fourier discrète, pour calculer la transformée de Fourier pour une liste de N valeurs, il suffit que de calculer la moité des termes

    N = len(x)

    if N == 1: #on divise jusqu'à ce que les parties soient de taille 1

        return x

    else:

        #la récursivité va ici pemettre de diviser la somme en plusieurs parties puis de remonter l'algorithme
        X_pair = FFT(x[::2])
        X_impair = FFT(x[1::2])
        facteur = np.exp(-2j*np.pi*np.arange(N)/ N)

        X = np.concatenate([X_pair+facteur[:int(N/2)]*X_impair,
             X_pair+facteur[int(N/2):]*X_impair])

        return X

def FFT2(x) :

    '''

    entrée : list
    sortie : list

    but : renvoie la transformée de Fourier de x, version lisible

    '''

    N = len(x)

    if N == 1 : #on divise la liste jusqu'à avoir une liste de taille 1
        return x

    else :
        X_pair = FFT2(x[::2])
        X_impair = FFT2(x[1::2])

        facteurs = [] #on créée nos facteurs
        for i in range (N) :
            facteurs.append(np.exp(-2j*np.pi*i/N))

        X1 = []
        X2 = []
        for i in range (len(X_pair)) :
            X1.append(X_pair[i]+facteurs[:int(N/2)][i]*X_impair[i])
            X2.append(X_pair[i]+facteurs[int(N/2):][i]*X_impair[i])

        return X1 + X2

def ToutesTFD (L) :

    '''

    entrée : list(list)
    sortie : list

    but : renvoie la transformée de Fourier discrète appliquée à chaque liste dans la liste d'entrée

    '''

    TFDS = []
    for x in L :
        tf = FFT2(Rajoute_0(x)) #transformée de Fourier appliquée à une liste
        TFDS.append(tf)

    return TFDS

def TFD(x) :

    '''

    entrée : list
    sortie : list

    but : renvoie la transformée de Fourier de x (version lente)

    '''

    N = len(x)

    Coefs = []
    for r in range (N) :
        Co = 0
        for k in range (N) :
            Co += x[k]*np.exp(-2j*np.pi*r*k/N)
        Coefs.append(Co)

    return Coefs

def ToutesTFDlentes (L) :

    '''

    entrée : list(list)
    sortie : list

    but : renvoie la transformée de Fourier discrète appliquée à chaque liste dans la liste d'entrée

    '''

    TFDS = []
    for x in L :
        tf = TFD(x) #transformée de Fourier appliquée à une liste
        TFDS.append(tf)

    return TFDS

# Périodogramme

def Per(L) :

    '''

    entrée : list(list)
    sortie : list(list)

    but : renvoie le périodogramme de chaque sous-liste de la liste d'entrée

    '''

    # On calcule le périodogramme pour toutes les sous listes, c'est-à-dire qu'on calcule le module de chaque valeur au carré, et on divise cette valeur par la taille de la sous-liste associée

    Pe = [] #liste des périodogrammes

    for x in L :
        PP = []
        for i in range (len(x)) :
            p = (1/len(x))*((np.abs(x[i]))**2)
            PP.append(p)
        Pe.append(PP)

    return Pe

# Mel filterbank

def Mel(f) :

    '''

    entrée : float
    sortie : float

    but : revoie dans le domaine de fréquences Mel la fréquence f

    '''

    return 1125*np.log(1+(f/700))

def Melinv(m) :

    '''

    entrée : float
    sortie : float

    but : revoie dans le domaine fréquenciel classique la fréquence f (Mel)

    '''

    return 700*(np.exp(m/1125)-1)

def Melfilterbank(L,n,sr) :

    '''

    entrée : list,int,float
    sortie : list(list)

    but : renvoie la liste des filtres du melfilterbank

    '''

    #n le nombre de filtres

    Fmin = 0
    Fmax = len(L)
    Mmin = Mel(Fmin)
    Mmax = Mel(Fmax)

    M = [] #liste de points entre Mmin et Mmax
    d = Mmax - Mmin
    for i in range (n+2) :
        M.append(Mmin + (i)*(d/(n+1)) )

    H = [] #même chose convertie en Hertz
    for x in M :
        H.append(Melinv(x))

    filterbank = []
    for m in range (n) :
        Temporary = [] #liste temporaire des points d'un filtre
        for k in range (int(H[-1])) : #création des filtres triangulaires
            if k < H[m] :
                Temporary.append(0)
            elif k >= H[m] and k < H[m+1] :
                Temporary.append((k-H[m])/(H[m+1]-H[m]))
            elif k == H[m+1] :
                Temporary.append(1)
            elif k > H[m+1] and k <= H[m+2] :
                Temporary.append((H[m+2]-k)/(H[m+2]-H[m+1]))
            elif k > H[m+2] :
                Temporary.append(0)
        filterbank.append(Temporary)

    return filterbank

def ValeursMaxListes(L) :

    '''

    entrée : list(list)
    sortie : list

    but : renvoie la liste constituée des valeurs maximales entre chaque liste

    '''

    #on vérifie que les listes de sont de la même taille
    liste_tailles = []
    for i in range (len(L)) :
        liste_tailles.append(len(L[i]))
    for x in liste_tailles :
        if liste_tailles[0] != x : #dès qu'une taille de liste est différente de la première, on revoie rien, en disant que les listes ne sont pas de la même taille
            print("les listes ne sont pas de la même taille")
            return None

    liste_valeursmax = []
    for k in range (len(L[0])) :
        ValeursIndexK = [] #on créé la liste des valeurs d'indice k
        for i in range (len(L)) :
            ValeursIndexK.append(L[i][k])
        liste_valeursmax.append(max(ValeursIndexK)) #on ajoute à la liste finale le maximum des valeurs d'indice k

    return liste_valeursmax

def Melfiltrage(L,n,sr) :

    '''

    entrée : list(list),int,float
    sortie : list(list)

    but : renvoie la liste des sous-listes de la liste d'entrée filtrées par le melfilterbank

    '''

    #on créée nos filtres

    Filtres = []
    for i in range (len(L)) :
        Filtres.append(Melfilterbank(L[i],n,sr))

    #on prend les valeurs maximales des filtres pour chaque trame, pour pouvoir filtrer chaque trame

    Filtres_max = []
    for i in range (len(Filtres)) :
        Filtres_max.append(ValeursMaxListes(Filtres[i]))

    #on filtre, c'est-à-dire qu'on multiplie chaque fréquence de chaque trame par la valeur correspondante du filtre associé (valeur entre 0 et 1)

    Listes_filtered = []
    for i in range (len(L)) :
        Liste_filtered_i = []
        for j in range (len(L[i])-1) :
            Liste_filtered_i.append(L[i][j]*Filtres_max[i][j])
        Listes_filtered.append(Liste_filtered_i)

    return Listes_filtered

# Logarithme

def Amplification(L) :

    '''

    entrée : list(list),int
    sortie : list(list)

    but : renvoie la liste des sous-listes de la liste d'entrée amplifiées

    '''

    #le signal ayant des valeurs trop petites, le logarithme renvoie une valeur négative, on amplifie donc le signal

    Liste_min = []
    for x in L :
        Liste_min.append(min(x))

    if min(Liste_min) >= 1 :

        return L

    else :
        Amp = 1 - min(Liste_min)

        L_ampli = [] #on les amplifie
        for i in range (len(L)) :
            L_ampli_i = []
            for j in range (len(L[i])) :
                L_ampli_i.append(Amp + L[i][j])
            L_ampli.append(L_ampli_i)

        return L_ampli

def PassageLog (L) :

    '''

    entrée : list(list)
    sortie : list(list)

    but : renvoie la liste des sous-listes de la liste d'entrée passées au logarithme

    '''

    #on passe au logarithme une liste de liste

    L_log = []
    for i in range (len(L)) :
        L_temp_log = []
        for j in range (len(L[i])) :
            if L[i][j] != 0 :
                L_temp_log.append(np.log(L[i][j]))
        L_log.append(L_temp_log)

    return L_log

# Transformée en cosinus discrète

def DCT(x) :

    '''

    entrée : list
    sortie : list

    but : renvoie la transformée en cosinus discrète de x (version lente)

    '''

    N = len(x) #taille de la liste

    Coefs = []
    for r in range (N) :
        Co = 0
        for k in range (N) :
            Co += x[k]*np.cos(np.pi*r*(k+1/2)/N)
        Coefs.append(Co)

    return Coefs

def FDCT(x) :

    '''

    entrée : list
    sortie : list

    but : renvoie la transformée en cosinus discrète de x, version rapide

    '''

    N = len(x) #taille

    if N == 1 :
        return x

    else :

        #on divise la liste
        X1 = []
        X2 = []
        for i in range (int(N/2)) :

            X1.append(x[i]+x[-i-1])
            X2.append((x[i]-x[-i-1])/(np.cos((i+0.5)*np.pi/N)*2))

        X1 = FDCT(X1)
        X2 = FDCT(X2)

        X = []
        for i in range (int(N/2)-1) :
            X.append(X1[i])
            X.append(X2[i] + X2[i+1])
        X.append(X1[-1])
        X.append(X2[-1])

        return X


def ToutesDCT(L) :

    '''

    entrée : list(list)
    sortie : list(list)

    but : renvoie la liste des transformées en cosinus discrètes des sous-listes de la liste d'entrée (rapide)

    '''

    L_nouveau = []

    for x in L :
        dct = FDCT(Rajoute_0(x))
        L_nouveau.append(dct)

    return L_nouveau

def ToutesDCTlentes(L) :

    '''

    entrée : list(list)
    sortie : list(list)

    but : renvoie la liste des transformées en cosinus discrètes des sous-listes de la liste d'entrée (lente)

    '''

    L_nouveau = []

    for x in L :
        dct = DCT(Rajoute_0(x))
        L_nouveau.append(dct)

    return L_nouveau

# Coefficients

def MFCC(L,c) :

    '''

    entrée : list(list)
    sortie : list(list)

    but : renvoie et affiche les c premiers mfcc en foncton du temps

    '''

    Liste_coefs = [] #on créée la liste des coefficients en fonction du temps
    for i in range (c) :
        Liste_coef_c = []
        for j in range (len(L)) :
            Liste_coef_c.append(L[j][i])
        Liste_coefs.append(Liste_coef_c)

    plt.imshow(Liste_coefs,origin = 'lower',cmap='jet',aspect="auto")
    plt.colorbar(label="Valeur coefficients")
    plt.gcf().set_size_inches(plt.gcf().get_size_inches()*1.5)
    plt.xlabel("Fenêtre",fontsize='xx-large')
    plt.ylabel("Numéro Coefficient",fontsize='xx-large')
    plt.show()

    return Liste_coefs

# Exécution

def Exe(f,c,e) :

    '''

    entrée : str,int,int
    sortie : list(list))

    but : execution du programme, f étant l'entrée, c le nombre de coefficients et e le nombre de points par période pour l'échantillonnage

    '''

    L = OuvrEtConv(f)
    if L != [] :
        LE = Echan(L,e)
        print("signal échantilloné à ", len(LE),"éléments au lieu de ", len(L))
        LN = SiNor(LE)
        print("signal normalisé")
        LA = SiAcc(LE,LN)
        print("signal pré accentué")
        LT = Trame(L,LA)
        print("signal séparé")
        TFDS = ToutesTFD(LT)
        print("transformés calculées")
        Pe = Per(TFDS)
        print("périodogramme calculé")
        M = Melfiltrage(Pe,10,44150)
        print("fréquences filtrées")
        Log = PassageLog(Amplification(M))
        print("signal passé au logarithme")
        dct = ToutesDCT(Log)
        print("transformé en cos")
        mfcc = MFCC(dct,c)
        print("coefficients calculés")

    return mfcc

### Programme Annexe

# Test log

def test_log(L) :

    '''

    entrée : list(list)
    sortie : int,int

    but : renvoie le nombre de valeurs interdites pour le logarithme

    '''

    zeros = 0
    negatifs = 0

    for sous_liste in L :
        for x in sous_liste :
            if x == 0 :
                zeros += 1
            if x < 0 :
                negatifs += 1

    return zeros,negatifs

# Temps comparatif

def ListeMoins (L,d) :

    '''

    entrée : list,int
    sortie : list

    but : renvoie la liste L avec les d derniers termes enlevés

    '''

    if d < len(L) :
        return L[0:len(L)-d]
    else :
        return L

def ExeC (E,L) :

    '''

    entrée : bool,list

    but : execution du programme avec ou sans fft

    '''

    if E == True : #execution avec fft
        if L!= [] :
            LN = SiNor(L)
            LA = SiAcc(L,LN)
            LT = Trame(L,LA)
            TFDS = ToutesTFD(LT)
            Pe = Per(TFDS)
            M = Melfiltrage(Pe,10,44150)
            Log = PassageLog(M)
            dct = ToutesDCT(Log)


    else : #execution sans fft
        if L!= [] :
            LN = SiNor(L)
            LA = SiAcc(L,LN)
            LT = Trame(L,LA)
            TFDS = ToutesTFDlentes(LT)
            Pe = Per(TFDS)
            M = Melfiltrage(Pe,10,44150)
            Log = PassageLog(M)
            dct = ToutesDCT(Log)

def ExeE (E,L) :

    '''

    entrée : bool,list

    but : execution du programme avec ou sans échantillonnage

    '''

    if E == True : #execution avec échantionnage
        if L!= [] :
            LE = Echan(L,4)
            LN = SiNor(LE)
            LA = SiAcc(LE,LN)
            LT = Trame(L,LA)
            TFDS = ToutesTFD(LT)
            Pe = Per(TFDS)
            M = Melfiltrage(Pe,10,44150)
            Log = PassageLog(M)
            dct = ToutesDCT(Log)


    else : #execution sans échantillonage
        if L!= [] :
            LN = SiNor(L)
            LA = SiAcc(L,LN)
            LT = Trame(L,LA)
            TFDS = ToutesTFD(LT)
            Pe = Per(TFDS)
            M = Melfiltrage(Pe,10,44150)
            Log = PassageLog(M)
            dct = ToutesDCT(Log)

def ExeTfft(E,L) :

    '''

    entrée : bool,list

    but : exécution de la tfd ou de la fft

    '''

    if E == False : #tfd
        if L!=[] :
            LN = SiNor(L)
            LA = SiAcc(L,LN)
            fft = TFD(LA)
    else : #fft
        if L!=[] :
            LN = SiNor(L)
            LA = SiAcc(L,LN)
            fft = FFT2(Rajoute_0(LA))

def Comparaisonfft_sanstrames (f,n) :

    '''

    entrée : str,int

    but : créer un graphe comparatif des méthodes fft et tfd

    '''

    LtempsE = [] #liste des temps avec fft
    LtempsS = [] #liste des temps sans fft
    Ltaille = [] #liste des tailles de liste
    L = OuvrEtConv(f)

    for i in range (n) :
        d = int(len(L)/n)*i #distance à retirer chaque fois
        LMod = ListeMoins(L,d) #liste avec cette distance en moins

        Ltaille.append(len(LMod)) #on ajoute à la liste des abscisses la taille de la liste

        s1 = time.time() #temps d'execution avec échantillion
        ExeTfft(True,LMod)
        LtempsE.append(time.time() - s1)
        print("Bon pour E", i+1, "fois")

        s2 = time.time() #temps d'execution sans échantillion
        ExeTfft(False,LMod)
        LtempsS.append(time.time() - s2)
        print("Bon pour S", i+1, "fois")

    plt.plot(Ltaille,LtempsE)
    plt.plot(Ltaille,LtempsS)
    plt.legend(['Transformée de Fourier rapide','Décomposition en série de Fourier finie'],fontsize='xx-large')
    plt.title("Temps comparatif de calcul de la transformée de Fourier rapide et discrète",fontsize='xx-large')
    plt.xlabel("Taille de liste (n)",fontsize='xx-large')
    plt.ylabel("Temps d'exécution (s)",fontsize='xx-large')
    plt.show()

def Comparaisonfft (f,n) :

    '''

    entrée : str,int

    but : créer un graphe comparatif des méthodes fft et tfd

    '''

    LtempsE = [] #liste des temps avec fft
    LtempsS = [] #liste des temps sans fft
    Ltaille = [] #liste des tailles de liste
    L = OuvrEtConv(f)

    for i in range (n) :
        d = int(len(L)/n)*i #distance à retirer chaque fois
        LMod = ListeMoins(L,d) #liste avec cette distance en moins

        Ltaille.append(len(LMod)) #on ajoute à la liste des abscisses la taille de la liste

        s1 = time.time() #temps d'execution avec échantillion
        ExeC(True,LMod)
        LtempsE.append(time.time() - s1)
        print("Bon pour E", i+1, "fois")

        s2 = time.time() #temps d'execution sans échantillion
        ExeC(False,LMod)
        LtempsS.append(time.time() - s2)
        print("Bon pour S", i+1, "fois")

    plt.plot(Ltaille,LtempsE)
    plt.plot(Ltaille,LtempsS)
    plt.legend(['Transformée de Fourier rapide','Décomposition en série de Fourier finie'],fontsize='xx-large')
    plt.title("Temps comparatif de calcul avec et sans transformée rapide",fontsize='xx-large')
    plt.xlabel("Taille de liste (n)",fontsize='xx-large')
    plt.ylabel("Temps d'exécution (s)",fontsize='xx-large')
    plt.show()

def ComparaisonE (f,n) :

    '''

    entrée : str,int

    but : créer un graphe comparatif des méthodes avec échantillonnage et sans

    '''

    LtempsE = [] #liste des temps avec échantillonnage
    LtempsS = [] #liste des temps sans échantillonnage
    Ltaille = [] #liste des tailles de liste
    L = OuvrEtConv(f)

    for i in range (n) :
        d = int(len(L)/n)*i #distance à retirer chaque fois
        LMod = ListeMoins(L,d) #liste avec cette distance en moins

        Ltaille.append(len(LMod)) #on ajoute à la liste des abscisses la taille de la liste

        s1 = time.time() #temps d'execution avec échantillion
        ExeE(True,LMod)
        LtempsE.append(time.time() - s1)
        print("Bon pour E", i+1, "fois")

        s2 = time.time() #temps d'execution sans échantillion
        ExeE(False,LMod)
        LtempsS.append(time.time() - s2)
        print("Bon pour S", i+1, "fois")

    plt.plot(Ltaille,LtempsE)
    plt.plot(Ltaille,LtempsS)
    plt.legend(['échantillonné','non échantillonné'],fontsize='xx-large')
    plt.title("Temps comparatif de calcul avec et sans échantillonnage",fontsize='xx-large')
    plt.xlabel("Taille de liste (n)",fontsize='xx-large')
    plt.ylabel("Temps d'exécution (s)",fontsize='xx-large')
    plt.show()

# Extraction

def Extrac(L) :

    '''

    entrée : list(list)

    but : créer un fichier texte avec les coefficients de fréquence mel

    '''

    Fichier = open("Pigeon3.txt",'a')
    for x in L :
        for k in x :
            Fichier.write(str(k))
            Fichier.write('\n')
        Fichier.write('\n')
    Fichier.close()

# Complexité

def N_carre(n) :

    '''

    entrée : int

    but : simule une fonction de complexité n carré, analogue à la décomposition en série de Fourier finie

    '''

    Simu = []
    for i in range (n) :
        Valeur = 0
        for k in range (n) :
            Valeur += k*np.exp(-2j*np.pi*k*i)
        Simu.append(Valeur)

def N_logN(n) :

    '''

    entrée : int

    but : simule une fonction de complexité n*log(n), analogue à la transformée de Fourier rapide

    '''

    Simu = []
    for i in range (n) :
        Valeur = 0
        for k in range (int(np.log(n))) :
            Valeur += k*np.exp(-2j*np.pi*k*i)
        Simu.append(Valeur)

def Liste_entiers(n) :

    '''

    entrée : int
    sortie : list

    but : renvoie la liste des entiers de 1 à n

    '''

    Liste = []
    for i in range (n) :
        Liste.append(i+1)

    return Liste

def ComparaisonComp(N,n) :

    '''

    entrée : int,int

    but : affiche un graphique comparatif entre les temps d'exécution de fonctions ayant pour complexité N² et Nlog(N)

    '''

    L = Liste_entiers(N)
    Ltaille = []
    LtempsC = []
    LtempsL = []

    for i in range (n) :
        d = int(len(L)/n)*i #distance à retirer chaque fois
        LMod = ListeMoins(L,d) #liste avec cette distance en moins

        Ltaille.append(len(LMod)) #on ajoute à la liste des abscisses la taille de la liste

        s1 = time.time() #temps d'execution de complexité en carré
        N_carre(len(LMod))
        LtempsC.append(time.time() - s1)
        print("Bon pour C", i+1, "fois")

        s2 = time.time() #temps d'execution de complexité en nlog(n)
        N_logN(len(LMod))
        LtempsL.append(time.time() - s2)
        print("Bon pour L", i+1, "fois")


    plt.plot(Ltaille,LtempsL)
    plt.plot(Ltaille,LtempsC)
    plt.legend(['Nlog(N)','N carré'],fontsize='xx-large')
    plt.title("Temps comparatif de complexité N carré et Nlog(N)",fontsize='xx-large')
    plt.xlabel("Taille de liste (N)",fontsize='xx-large')
    plt.ylabel("Temps d'exécution (s)",fontsize='xx-large')
    plt.show()

# Echelle de Mel

def Echelle_Mel(Hz) :

    '''

    entrée : int

    but : affiche le graphe de la perception auditive

    '''

    Liste_Mel = []
    Liste_Hz = Liste_entiers(int(Hz))

    for x in Liste_Hz :
        Liste_Mel.append(Mel(x))

    plt.plot(Liste_Hz,Liste_Mel)
    plt.title("Perception auditive",fontsize='xx-large')
    plt.xlabel("Hertz (Hz)",fontsize='xx-large')
    plt.ylabel("Mel (M)",fontsize='xx-large')
    plt.show()