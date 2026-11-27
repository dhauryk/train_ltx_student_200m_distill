# CelebV-HQ setup

CelebV-HQ is restricted to non-commercial research. Read and accept the
[official agreement](https://celebv-hq.github.io/) before downloading.

## Option 1: download and process from source

Install the project environment and make sure `ffmpeg`/`ffprobe` are on `PATH`.
YouTube also requires Node.js or Deno in that same environment so yt-dlp can
solve the player challenge:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg nodejs
```

Then run:

```bash
python download_celebv_hq.py --accept-license
```

If YouTube asks to sign in, pass cookies from the browser where you are logged in.

On native Linux, close the browser and run:

```bash
python download_celebv_hq.py --accept-license --cookies-from-browser firefox
```

On WSL2 the Windows browser is not visible that way. Chrome and Edge cookies are
encrypted with Windows DPAPI, so export a Netscape `cookies.txt` in Windows and
pass its `/mnt/c` path:

```bash
python download_celebv_hq.py --accept-license --cookies /mnt/c/Users/<WindowsUser>/Downloads/cookies.txt
```

A Windows Firefox profile can be read directly while Firefox is closed:

```bash
python download_celebv_hq.py --accept-license \
  --cookies-from-browser "firefox:/mnt/c/Users/<WindowsUser>/AppData/Roaming/Mozilla/Firefox/Profiles/<profile>"
```

The script downloads the official metadata, groups clips by YouTube source,
processes all available clips to face-centric 512×512 MP4 files, and writes:

```text
celebv_hq/
  celebv_hq_info.json
  celebv_hq_manifest.csv
  videos/
```

It deletes each raw source after its clips are processed. Add `--keep-raw` if
you have enough disk space.

## Option 2: index an already processed copy

Place official processed clips under `celebv_hq/videos/` using their current
CelebV-HQ clip IDs, then run:

```bash
python download_celebv_hq.py --index-only --accept-license
```

Legacy v0.3 `deprecated_name` files are renamed automatically when metadata
provides the mapping.

## Training behavior

No CelebV-HQ clip is removed from training:

- every action category, including talking, singing, eating, laughing, and
  expressive motion, can contribute real-video flow/x0 supervision;
- only technically unsuitable clips (too short, low-FPS, blurry, or masked)
  are always routed to image-conditioned teacher rollout instead of being
  removed; compatible clips are split between real-video and rollout batches by
  the 40–50% curriculum;
- rollout uses the clip caption in 65% of samples and a motion/camera prompt
  bank with combined wind+camera scenarios in 35%;
- rollout probability grows continuously from 40% to 50%; windows cover the
  available trajectory uniformly at CFG 3.0, and their last three steps are
  differentiable against same-state teacher corrections;
- checkpoint selection includes decoded LPIPS together with teacher-relative
  endpoint, temporal, and motion metrics.

The trainer uses every manifest clip and writes new checkpoints to
`outputs_ltx_student_celebv_hq/` (14×2048 student).

Regenerate the deployment prompt embeddings once after this migration:

```bash
python save_portrait_embeds.py
```

`preview_prompts.py` contains only the user-facing generation prompts.
Training uses CelebV-HQ action captions plus the prompt bank from
`ltx_portrait_student.py`. Validation uses the deterministic captions of its
CelebV-HQ clips, so changing preview wording does not require retraining.

## WSL2: рабочая последовательность

Команды запускать в WSL из `~/studies4`. Активный Python 3.13 Poetry не
использует: проект требует `>=3.10,<3.13`, и Poetry сам берёт 3.12.
Повторный запуск скачивания пропускает уже готовые клипы.

### 1. Окружение

`ffmpeg` и `ffprobe` должны быть в `PATH`.

```bash
poetry install
poetry run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ffmpeg -version
ffprobe -version
```

### 2. Deno без apt

YouTube отдаёт только превью, пока `yt-dlp` не решит JS-challenge. Нужен Deno
2.3+ и скрипты EJS. Скрипт скачивания передаёт `--js-runtimes` и
`--remote-components ejs:github` сам.

Если `apt` занят или нет `unzip`, распаковать релиз Deno через Python:

```bash
arch=$(uname -m)
case "$arch" in
  x86_64) asset=deno-x86_64-unknown-linux-gnu.zip ;;
  aarch64) asset=deno-aarch64-unknown-linux-gnu.zip ;;
  *) echo "unsupported $arch"; exit 1 ;;
esac
curl -fL -o /tmp/deno.zip "https://github.com/denoland/deno/releases/latest/download/$asset"
mkdir -p "$HOME/.deno/bin"
poetry run python -c "
import zipfile
from pathlib import Path
dest = Path.home() / '.deno' / 'bin'
dest.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile('/tmp/deno.zip') as zf:
    zf.extractall(dest)
(dest / 'deno').chmod(0o755)
"
grep -q '.deno/bin' ~/.bashrc || echo 'export PATH="$HOME/.deno/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.deno/bin:$PATH"
deno --version
```

### 3. Cookies YouTube в Windows

В WSL `--cookies-from-browser` не видит браузер Windows, а cookies Chrome и
Edge зашифрованы через DPAPI.

1. В Chrome или Edge включить у расширения **Get cookies.txt LOCALLY** доступ в инкогнито и ко всем сайтам. Старое расширение **Get cookies.txt** не использовать.
2. Открыть окно инкогнито, войти в YouTube и в той же вкладке открыть `https://www.youtube.com/robots.txt`.
3. Нажать значок пазла, открыть **Get cookies.txt LOCALLY**, выбрать формат **Netscape** и нажать **Export**. **Export All Cookies** не нажимать. На самой странице `robots.txt` кнопки экспорта нет.
4. Закрыть окно инкогнито. Файл появляется в загрузках Windows, обычно как `www.youtube.com_cookies.txt`.

Проверка из WSL:

```bash
head -n 1 /mnt/c/Users/USER/Downloads/www.youtube.com_cookies.txt
```

Первая строка должна быть `# Netscape HTTP Cookie File`. Файл — это сессия
YouTube: не коммитить и не отправлять. Если снова появится
`Sign in to confirm you're not a bot`, экспортировать cookies заново.

### 4. Скачать и нарезать CelebV-HQ

```bash
export PATH="$HOME/.deno/bin:$PATH"
poetry run python download_celebv_hq.py --accept-license \
  --cookies /mnt/c/Users/USER/Downloads/www.youtube.com_cookies.txt
```

Отдельные ролики могут остаться недоступными. Скрипт пропускает такой source и
продолжает. Результат:

```text
celebv_hq/
  celebvhq_info.json
  celebv_hq_manifest.csv
  videos/
```

### 5. Preview embeds

После каждой правки `preview_prompts.py`:

```bash
poetry run python save_portrait_embeds.py
```

Эти prompts не участвуют в обучении. Обучение использует action captions
CelebV-HQ и небольшой motion/camera prompt bank из `ltx_portrait_student.py`.

### 6. Обучение

```bash
poetry run python -m ipykernel install --user --name=optimization-experiment --display-name="optimization-experiment"
poetry run jupyter notebook train_ltx_student_200m_distill.ipynb
```

В ноутбуке выбрать kernel `optimization-experiment` и выполнить все ячейки.
Веса пишутся в `outputs_ltx_student_celebv_hq`.

### 7. Preview

```bash
poetry run jupyter notebook generate_ltx_student_preview.ipynb
```

Тот же kernel. Сравнивать `student_best.mp4`, `student_best_ema.mp4` и
`teacher_reference.mp4`.