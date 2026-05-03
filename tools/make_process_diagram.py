#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "processo_video_youtube.jpg"

W, H = 1920, 1080
BG = (248, 250, 252)
HEADER = (15, 23, 42)
INK = (17, 24, 39)
MUTED = (75, 85, 99)
LINE = (148, 163, 184)
WHITE = (255, 255, 255)
PANEL = (241, 245, 249)
BLUE = (37, 99, 235)
GREEN = (22, 163, 74)
ORANGE = (234, 88, 12)
PURPLE = (124, 58, 237)
CYAN = (8, 145, 178)
PINK = (190, 24, 93)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE = font(56, True)
SUBTITLE = font(29)
CARD_TITLE = font(29, True)
CARD_TEXT = font(23)
BADGE = font(20, True)
SMALL = font(21)
STEP = font(24, True)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fnt, fill=INK, anchor=None) -> None:
    draw.text(xy, value, font=fnt, fill=fill, anchor=anchor)


def wrap_to_width(draw: ImageDraw.ImageDraw, value: str, fnt, max_width: int) -> list[str]:
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=fnt) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, fnt, fill, max_width: int, max_lines: int) -> None:
    lines = wrap_to_width(draw, value, fnt, max_width)[:max_lines]
    yy = y
    for line in lines:
        draw.text((x, yy), line, font=fnt, fill=fill)
        yy += fnt.size + 6


def rounded(draw: ImageDraw.ImageDraw, xy, fill, outline=LINE, width=3, radius=20) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def down_arrow(draw: ImageDraw.ImageDraw, x: int, y1: int, y2: int, color=LINE) -> None:
    draw.line((x, y1, x, y2), fill=color, width=5)
    draw.polygon([(x, y2), (x - 13, y2 - 18), (x + 13, y2 - 18)], fill=color)


def badge(draw: ImageDraw.ImageDraw, x: int, y: int, label: str) -> None:
    w = int(draw.textlength(label, font=BADGE)) + 30
    draw.rounded_rectangle((x, y, x + w, y + 34), radius=17, fill=(239, 246, 255), outline=(191, 219, 254), width=2)
    draw_text(draw, (x + 15, y + 6), label, BADGE, BLUE)


def process_card(
    draw: ImageDraw.ImageDraw,
    y: int,
    number: str,
    color,
    title: str,
    body: str,
    tool: str,
    fill=WHITE,
) -> None:
    x1, y1, x2, y2 = 120, y, 1340, y + 84
    rounded(draw, (x1, y1, x2, y2), fill, radius=18)
    draw.rounded_rectangle((x1, y1, x1 + 14, y2), radius=12, fill=color)

    if number:
        draw.ellipse((x1 + 35, y1 + 20, x1 + 79, y1 + 64), fill=color)
        draw_text(draw, (x1 + 57, y1 + 29), number, STEP, WHITE, anchor="ma")
        title_x = x1 + 105
    else:
        title_x = x1 + 35

    draw_text(draw, (title_x, y1 + 16), title, CARD_TITLE)
    draw_wrapped(draw, title_x, y1 + 50, body, CARD_TEXT, MUTED, 690, 1)
    badge(draw, 1135, y1 + 25, tool)


def side_panel(draw: ImageDraw.ImageDraw) -> None:
    rounded(draw, (1415, 235, 1800, 615), (240, 253, 244), outline=(134, 239, 172), radius=24)
    draw_text(draw, (1450, 270), "Saída final", font(32, True), INK)
    items = ["vídeo editado", "thumbnail", "youtube.md", "plano de cortes"]
    y = 335
    for item in items:
        draw.ellipse((1450, y + 7, 1468, y + 25), fill=GREEN)
        draw_text(draw, (1482, y), item, font(26, True), INK)
        y += 52
    draw_wrapped(draw, 1450, 535, "Pronto para publicar ou revisar no YouTube.", font(22), MUTED, 290, 2)

    rounded(draw, (1415, 650, 1800, 955), PANEL, outline=(203, 213, 225), radius=24)
    draw_text(draw, (1450, 685), "Ferramentas", font(32, True), INK)
    tools = ["Python", "FFmpeg", "faster-whisper", "OpenCV", "OpenAI API"]
    y = 740
    for tool in tools:
        badge(draw, 1450, y, tool)
        y += 42


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 0, W, 170), fill=HEADER)
    draw_text(draw, (80, 46), "Como 4 horas viram um vídeo de 10 minutos", TITLE, WHITE)
    draw_text(draw, (82, 116), "Pipeline automático para motovlog: cortes, IA, render e pacote do YouTube", SUBTITLE, (203, 213, 225))

    rows = [
        ("", ORANGE, "Entrada", "10 vídeos de 25 min vindos da câmera.", "arquivos MP4", (255, 247, 237)),
        ("1", BLUE, "Unir clipes", "Corta o trecho duplicado entre arquivos e cria um vídeo contínuo.", "FFmpeg", WHITE),
        ("2", GREEN, "Transcrever", "Extrai o áudio, reduz ruído de vento e transforma fala em texto.", "faster-whisper", WHITE),
        ("3", CYAN, "Analisar imagem", "Mede mudança de paisagem, cor, nitidez e exposição.", "OpenCV", WHITE),
        ("4", PURPLE, "Pontuar trechos", "Cria janelas candidatas e combina fala, visual e energia.", "Python", WHITE),
        ("5", PINK, "Escolher cortes", "A IA decide quais momentos contam melhor a história.", "OpenAI", WHITE),
        ("6", ORANGE, "Renderizar e exportar", "Aplica melhorias de imagem e áudio, corta e concatena.", "FFmpeg", WHITE),
    ]

    start_y = 220
    gap = 104
    card_h = 84
    for i, row in enumerate(rows):
        y = start_y + i * gap
        process_card(draw, y, *row)
        if i < len(rows) - 1:
            down_arrow(draw, 730, y + card_h + 6, y + gap - 7)

    side_panel(draw)

    rounded(draw, (120, 970, 1800, 1032), PANEL, outline=(203, 213, 225), radius=22)
    draw_text(draw, (155, 987), "Resumo", font(29, True), INK)
    summary = "Dados locais escolhem bons candidatos; a OpenAI ajuda na decisão editorial, título, descrição, tags e camada da thumbnail."
    draw_wrapped(draw, 285, 990, summary, SMALL, MUTED, 1320, 1)

    draw_text(draw, (80, 1048), "Moto Editor", SMALL, MUTED)
    draw_text(draw, (1840, 1048), "Gerado para explicar o processo no vídeo", SMALL, MUTED, anchor="ra")

    img.save(OUT, "JPEG", quality=94, subsampling=0)
    print(OUT)


if __name__ == "__main__":
    main()
