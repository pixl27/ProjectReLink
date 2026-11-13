# Project ReLink

**Zelda OOT Remake Recomp.** Une recompilation statique, non officielle, de *The Legend of Zelda: Ocarina of Time* (le remake
Switch 2) en jeu Windows natif : le code du jeu est traduit en code PC et compilé sur votre machine, sans émulateur.

[English below](#english)

![Le lanceur de Project ReLink](docs/launcher.png)

**Ce dépôt ne contient aucun fichier du jeu.** ReLink fabrique le jeu sur votre PC à partir de votre propre copie.

## Installer et jouer

1. Téléchargez `ReLink-Setup.exe` dans la [dernière version](https://github.com/pixl27/ProjectReLink/releases/latest).
2. Lancez-le. Il télécharge ReLink et ses outils (environ 500 Mo), les vérifie et ouvre le lanceur. Rien d'autre à installer :
   ReLink apporte son compilateur et son Python.
3. Choisissez votre copie du jeu. Le lanceur cherche les fichiers `.nup` dans Téléchargements, Bureau et Documents ; vous
   pouvez aussi en choisir un, le déposer sur la fenêtre, ou donner les dossiers `romfs` et `exefs`. La version est vérifiée.
4. Appuyez sur **Installer**. Comptez 10 à 45 minutes selon le processeur ; le PC reste utilisable.
5. Appuyez sur **Jouer**. Les raccourcis du bureau et du menu Démarrer ouvrent ensuite le lanceur.

![L'installation](docs/install.png)

Tant qu'aucune version n'apparaît sur la page des versions, l'installateur n'est pas encore publié : voir
[Construire depuis les sources](#construire-depuis-les-sources).

### Ce qu'il vous faut

- **Votre copie du jeu, dans la version prise en charge** : programme `0400486024DEC000`, mise à jour v131072, build
  `e0f3ccfa`. Les autres versions ne fonctionnent pas. Un fichier `.nup`, ou les dossiers `romfs` et `exefs`.
- Windows 10 ou 11, 64 bits, et un processeur avec AVX2.
- Une carte graphique avec Vulkan 1.3 et les mesh shaders, pilote à jour : NVIDIA GeForce RTX 20 ou plus récente, AMD Radeon
  RX 6000 ou plus récente, Intel Arc. Nous n'avons testé que sur NVIDIA.
- 8 Go de mémoire vidéo conseillés, 16 Go de RAM ou plus.
- Environ 60 Go libres pendant l'installation, et une connexion pour le téléchargement.

### Mettre à jour

Le lanceur dit quand une nouvelle version est sortie. **Mettre à jour** ne télécharge que ce qui a changé et reconstruit le
jeu dans le même dossier à partir des fichiers du jeu qu'il garde : votre `.nup` n'est plus nécessaire. Les sauvegardes
(`user`), les réglages et les mods sont conservés.

### Où vont les fichiers

- Le jeu : le dossier que vous choisissez (`Games\ReLink` dans votre dossier personnel par défaut). Votre copie n'est jamais
  modifiée.
- ReLink lui-même : `%LOCALAPPDATA%\ReLink`. Les fichiers de construction (environ 4 Go) sont supprimés quand le jeu est prêt.
- Pour désinstaller : supprimez le dossier du jeu, `%LOCALAPPDATA%\ReLink` et les raccourcis.

## Réglages

Le lanceur règle l'affichage (plein écran, résolution du rendu 3D, mise à l'échelle, netteté, ultra-large), la fluidité
(60 images par seconde ou débloqué, génération d'images), la langue du jeu et les contrôles. Il peut passer l'écran à 60 Hz le
temps de la partie, pour que chaque image dure le même temps sur un écran à 100 ou 144 Hz, puis remet la fréquence d'origine.
En jeu, **Inser** ou **F1** ouvre le même menu ; **F11** bascule le plein écran.

Manette Xbox ou toute manette XInput ; clavier et souris aussi (la liste des touches est dans l'onglet Contrôles).

## Images par seconde mesurées

Sur notre configuration : **Ryzen 5 5500, GeForce RTX 3060 Ti (8 Go), 16 Go de RAM, Windows 11**. Jeu limité à 60 images par
seconde, rendu 3D à 75 %, version 0.6.1. Ce sont des moyennes sur des fenêtres de 30 à 60 secondes.

| Lieu | Images par seconde |
|---|---|
| Plaine d'Hyrule, en marchant | 59,8 |
| Village Cocorico, en marchant | 59,0 |
| Château d'Hyrule, en marchant | 58,8 |
| Bois perdus, en marchant | 59,6 |
| Bosquet sacré, en marchant | 60,0 |
| Lac Hylia, en marchant | 59,2 |
| Fontaine Zora, en marchant | 58,3 |
| Place du marché, en marchant | 54,1 |
| Arbre Mojo, caverne Dodongo, puits (enfant) | 60 |
| Ventre de Jabu-Jabu (enfant) | 58 à 60 |
| Village Cocorico, temple du feu, forteresse Gerudo (adulte, à l'arrêt) | 60 |
| Temple de l'eau (adulte, à l'arrêt) | 58,0 |

En rendu natif (100 %) : 51 à 53 dans les passages les plus chargés du village Kokiri, 53,5 à 56,7 dans la forêt, 52,8 à la
fontaine Zora. À la fontaine Zora, la résolution du rendu décide : automatique 54,5, 75 % 58,3, 67 % 58,6, 50 % 60,0.

Ce qui n'est pas encore atteint ni mesuré : le 60 constant partout (la place du marché et le rendu natif restent en dessous),
les combats de boss, les sessions de plus d'une demi-heure, et les cartes AMD et Intel.

## Si le jeu ralentit avec Link adulte : vérifier le dump

Certains dumps contiennent quelques textures abîmées. Le jeu recharge alors sans fin le modèle qui en a besoin, ce qui occupe
plus d'un cœur du processeur en permanence dès que Link est adulte. Deux outils du dépôt s'en occupent, avec vos propres
fichiers (rien du jeu n'est fourni) :

```powershell
python -I tools/dump_check.py "C:\Games\ReLink\romfs\Tex"
python -I tools/dump_repair.py "C:\Games\ReLink\romfs" "C:\Games\ReLink\mods\00-dump-repair"
```

Le premier compte les fichiers abîmés ; le second écrit un petit mod qui remplace ceux qu'il connaît par une texture voisine
intacte. Ils demandent le paquet `zstandard`, que le Python de ReLink possède
(`%LOCALAPPDATA%\ReLink\versions\<version>\ReLink\tools\python\python.exe`).

## Mods

Les mods RomFS vont dans `mods\<nom>\romfs\...` dans le dossier du jeu, avec les mêmes chemins que le jeu. Le journal
(`hookshot_log.txt`) nomme chaque mod lu. Les patchs du code (`exefs`) ne sont pas pris en charge : le code est compilé à
l'avance.

## Construire depuis les sources

Pour les développeurs. Le lanceur exécute le même script, `nx2/make_game.py`.

```powershell
python -I nx2/make_game.py --nup "C:\chemin\votre-jeu.nup" --out "C:\Games\ReLink" --language fr
python -I nx2/make_game.py --romfs "C:\chemin\romfs" --exefs "C:\chemin\exefs" --out "C:\Games\ReLink" --language fr
```

`--check` avec votre `.nup` vérifie seulement la version. `--log fichier` garde la sortie. `HOOKSHOT_JOBS=<n>` limite le nombre
de fichiers compilés à la fois.

Il faut :

- Python 3 de python.org, avec `pip install lz4 numpy capstone "unicorn>=2.1,<2.2" zstandard`.
- [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) (la version `ucrt-x86_64`), décompressé en `llvm-mingw` dans le clone,
  ou `HOOKSHOT_MINGW` pointant sur son dossier.
- Les en-têtes Vulkan et FFmpeg dans la disposition du cache CPM, avec `RX_CPM` pointant sur leur dossier. Sans eux, le
  rendu Vulkan n'est pas construit.
- Les fichiers précompilés de `third_party` et le profil de jeu de `work`. Un ReLink installé les contient, dans
  `%LOCALAPPDATA%\ReLink\versions\<version>\ReLink\tools`.

Le lanceur est dans `launcher/` (`dotnet build`, .NET 9), le petit installateur dans `launcher/setup/`. La publication d'une
version est décrite dans [docs/releasing.md](docs/releasing.md).

### Comment ça marche

1. `nx2/lift.py` traduit le code ARM64 du jeu en C.
2. `nx2/build.py` compile ce C avec `host/`, qui tient le rôle du système de la console.
3. `host/nvn_native.c` enregistre les appels graphiques du jeu, et `gpu/vk/` les rejoue sur Vulkan.

## Licence

Project ReLink est un logiciel libre sous la [licence publique générale GNU, version 3 ou ultérieure](LICENSE)
(GPL-3.0-or-later). Il reprend et modifie du code publié sous GPL par d'autres projets (modifications de novembre 2026) ; les
mentions de droits d'auteur et les licences de chaque partie sont conservées dans leurs fichiers et leurs dossiers.

Project ReLink est un projet de fans non officiel, sans lien avec Nintendo. The Legend of Zelda et Ocarina of Time sont des
marques de Nintendo.

---

## English

**Project ReLink** is an unofficial static recompilation of *The Legend of Zelda: Ocarina of Time* (the Switch 2 remake) into
a native Windows game: the game's code is translated to PC code and compiled on your machine, with no emulator.
**No game files are in this repository**: ReLink builds the game on your PC from your own copy.

### Install and play

1. Download `ReLink-Setup.exe` from the [latest release](https://github.com/pixl27/ProjectReLink/releases/latest).
2. Run it. It downloads ReLink and its tools (about 500 MB), checks them and opens the launcher. Nothing else to install.
3. Pick your copy of the game: a `.nup` file (the launcher looks in Downloads, Desktop and Documents; you can also browse or
   drop it on the window), or the `romfs` and `exefs` folders. The version is checked.
4. Press **Install**. It takes 10 to 45 minutes depending on your CPU; you can keep using the PC.
5. Press **Play**. The launcher is in English when Windows is not in French.

Until a release shows on the releases page, the installer is not published yet: see "Construire depuis les sources" above
(the commands are the same in any language).

You need the supported game version (program `0400486024DEC000`, update v131072, build `e0f3ccfa`), 64-bit Windows 10 or 11, a
CPU with AVX2, a graphics card with Vulkan 1.3 and mesh shaders (GeForce RTX 20 series or newer, Radeon RX 6000 or newer, Intel
Arc; only NVIDIA was tested by us), 16 GB of RAM, and about 60 GB free while it installs.

The launcher says when a new release is out; **Update** downloads only what changed and rebuilds the game in place. Saves,
settings and mods are kept.

### Measured frame rates

On a Ryzen 5 5500, GeForce RTX 3060 Ti (8 GB), 16 GB of RAM, Windows 11, with the 60 fps cap and the 3D render resolution at
75 %: 58 to 60 fps in most places (Hyrule Field 59.8, Kakariko 59.0, Lake Hylia 59.2, Zora's Fountain 58.3; dungeons 58 to
60), 54.1 walking through the Market. A steady 60 everywhere is not reached yet, and boss fights, long sessions and AMD or
Intel cards have not been measured. The full table is in the French section.

### License

GPL-3.0-or-later ([LICENSE](LICENSE)). It reuses and modifies code published under the GPL by other projects (modified in
November 2026); each part keeps its copyright notices and licenses in its files and folders. Unofficial fan project, not
affiliated with Nintendo; The Legend of Zelda and Ocarina of Time are trademarks of Nintendo.
