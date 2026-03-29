![ISC Logo](http://iscracingteam.com/wp-content/uploads/2022/03/Picture5.jpg)

# IFS08 - DV_AMI

Firmware embebido para el **Autonomous Mission Indicator (AMI)** del IFS08, desarrollado en STM32H7 con micro-ROS.

---

## Primeros pasos

1. Crea una cuenta en GitHub si aún no tienes una.
2. Descarga e instala [GitHub Desktop](https://desktop.github.com/) (básico) o [Git CLI](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) (avanzado).

   - Si es la primera vez que utilizas GitHub Desktop, asegúrate de leer el [Manual del usuario](https://help.github.com/desktop/guides/).
   - Si estás usando Git por primera vez, empieza con un tutorial. Hay muchos disponibles en línea:
     - [Tutorial de Git](https://git-scm.com/docs/gittutorial)
     - [Atlassian Git Tutorial](https://www.atlassian.com/git/tutorials/)
   - Guarda una copia de [GitHub's Git Cheat Sheet](https://services.github.com/kit/downloads/github-git-cheat-sheet.pdf) como referencia.

3. Clona este repositorio en tu ordenador:
   - SSH: `git@github.com:isc-fs/IFS08-DV_AMI.git`
   - HTTPS: `https://github.com/isc-fs/IFS08-DV_AMI.git`

---

## Cómo trabajamos con este repositorio

### Ramas principales

El repositorio tiene dos ramas permanentes:

**`main`** es la rama de producción. Contiene únicamente código validado que puede flashearse en el coche. Nunca trabajes directamente sobre ella.

**`dev`** es la rama de desarrollo. Es el punto de integración donde converge el trabajo de todos. Tampoco trabajes directamente sobre ella — todos los cambios llegan a través de una rama de trabajo.

```
main  ──────────────────●──────────────────────●──▶  (solo releases validados)
                        ↑                      ↑
dev   ──────●───●───●───●───●───●───●───●───●──●──▶  (integración continua)
            ↑   ↑       ↑   ↑   ↑       ↑   ↑
          feat/1 fix/1 feat/2 fix/2   feat/3 fix/3
```

### Ramas de trabajo

Todo el trabajo — ya sea una nueva funcionalidad o una corrección de errores — se realiza en una **rama de trabajo** creada desde `dev`. Cuando el trabajo está listo, se abre un Pull Request hacia `dev`, se revisa, se fusiona y la rama se elimina.

Hay dos tipos de rama, cada uno con su propio contador numérico independiente:

```
feat/<n>   →  nueva funcionalidad  (feat/1, feat/2, feat/3 ...)
fix/<n>    →  corrección de error  (fix/1,  fix/2,  fix/3  ...)
```

Los contadores de `feat` y `fix` son independientes: `feat/2` y `fix/2` pueden existir al mismo tiempo sin conflicto.

### Seguimiento del historial de ramas

Las ramas se eliminan tras fusionarse para mantener el repositorio limpio. El historial de cada rama se conserva en **GitHub Issues**.

Cada rama tiene un issue asociado. El issue lleva una **etiqueta** (`feat` o `fix`) y su título incluye el número de la rama, por ejemplo: `[feat/3] Añadir broadcast CAN para el estado de misión`. Cuando la rama se fusiona y elimina, el issue se cierra — convirtiéndose en un registro permanente de todo el trabajo realizado.

Para ver qué ramas están activas: filtra los issues por etiqueta y estado `open`.
Para consultar el historial completo: filtra por etiqueta y estado `closed`.
El número para la siguiente rama de cada tipo es el último issue cerrado de ese tipo más uno.

> Ejemplo: si el último issue cerrado con etiqueta `feat` es `[feat/4] ...`, la siguiente rama de funcionalidad será `feat/5`.

---

## Automatización

El repositorio incluye un workflow de GitHub Actions que gestiona los issues de seguimiento de forma automática. No es necesario configurar nada — funciona para todos los desarrolladores desde el momento en que crean una rama.

### Creación automática del issue

Cuando se publica una rama `feat/*` o `fix/*` en GitHub, el workflow abre automáticamente un issue con:

- El título `[feat/N]` o `[fix/N]` correspondiente
- La etiqueta correcta (`feat` o `fix`)
- Una plantilla con secciones para describir el trabajo y añadir notas
- El nombre del desarrollador que creó la rama

### Aviso de número incorrecto

Si el número de la rama no es el siguiente esperado (ya sea demasiado bajo o demasiado alto), el issue mostrará un aviso indicando cuál es el número correcto y pidiendo que se recree la rama con el nombre adecuado.

### Descripción automática desde el primer commit

Cuando el desarrollador realiza su primer commit y lo publica, el workflow actualiza automáticamente la sección *"¿Qué hace esta rama?"* del issue con el mensaje de ese commit.

- Si el desarrollador edita el issue manualmente antes de hacer el primer push, el workflow no sobreescribirá la descripción.
- Solo se actualiza una vez — los commits posteriores no modifican el issue.

---

## Flujo de trabajo paso a paso

### 1. Crear la rama

```bash
# Asegúrate de estar en una dev actualizada
git checkout dev
git pull origin dev

# Crea tu rama usando el siguiente número disponible para su tipo
# (último issue cerrado de ese tipo + 1)
git checkout -b feat/5    # o fix/3, según el contador de ese tipo
```

> Para saber qué número usar: ve a **Issues → filtra por etiqueta `feat` o `fix` → ordena por más reciente** y lee el último número.

### 2. Publicar la rama

```bash
git push origin feat/5
```

El issue de seguimiento se abrirá automáticamente en GitHub en unos segundos.

### 3. Trabajar y hacer commits

```bash
# Realiza tus cambios y haz commit con un mensaje descriptivo
git add .
git commit -m "descripción clara de lo que hace este commit"

# Publica los cambios
git push origin feat/5
```

El mensaje de tu **primer commit** se usará automáticamente para rellenar la descripción del issue.

### 4. Abrir un Pull Request

Cuando el trabajo esté listo, abre un Pull Request en GitHub desde tu rama hacia `dev`. En la descripción del PR escribe `Closes #<número-de-issue>` para que el issue se cierre automáticamente al fusionarse.

Antes de solicitar la revisión, comprueba que:
- El código compila sin errores ni avisos
- Has probado el cambio en el banco si corresponde
- El PR apunta a `dev`, no a `main`

### 5. Revisión y fusión

Otro miembro del equipo revisará el PR. Una vez aprobado, se fusiona en `dev` y la rama se elimina. El issue quedará cerrado como registro permanente.

### 6. Fusión en main

Cuando `dev` tiene un conjunto de cambios validados listos para el coche, un responsable abre un Pull Request desde `dev` hacia `main`. Esto solo ocurre tras la validación completa del firmware (HIL/banco).

---

*ISC Racing Team — IFS08 Driverless*
