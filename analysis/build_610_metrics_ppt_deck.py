from __future__ import annotations

from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


OUTPUT_DIR = Path(r"G:\610\Mammo-CLIP\analysis\ppt_metric_tables_20260611")
DECK_PATH = OUTPUT_DIR / "mammoclip_metrics_tables_20260611.pptx"

SLIDE_IMAGES = [
    OUTPUT_DIR / "ppt_table_image.png",
    OUTPUT_DIR / "ppt_table_patient_id_mean.png",
    OUTPUT_DIR / "ppt_table_patient_id_laterality_max.png",
]

PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

EMU_WIDTH = 12192000
EMU_HEIGHT = 6858000


def ensure_inputs() -> None:
    if not DECK_PATH.exists():
        raise FileNotFoundError(f"Template PPTX not found: {DECK_PATH}")
    for image_path in SLIDE_IMAGES:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing slide image: {image_path}")


def build_slide_xml(embed_rid: str) -> bytes:
    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="2" name="Metrics Table"/>
          <p:cNvPicPr>
            <a:picLocks noChangeAspect="1"/>
          </p:cNvPicPr>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill>
          <a:blip r:embed="{embed_rid}"/>
          <a:stretch>
            <a:fillRect/>
          </a:stretch>
        </p:blipFill>
        <p:spPr>
          <a:xfrm>
            <a:off x="0" y="0"/>
            <a:ext cx="{EMU_WIDTH}" cy="{EMU_HEIGHT}"/>
          </a:xfrm>
          <a:prstGeom prst="rect">
            <a:avLst/>
          </a:prstGeom>
        </p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:sld>
"""
    return xml.encode("utf-8")


def patch_relationships(rels_bytes: bytes, image_target: str) -> tuple[bytes, str]:
    root = ET.fromstring(rels_bytes.decode("utf-8-sig"))
    relationship_tag = f"{{{PACKAGE_REL_NS}}}Relationship"
    existing_ids = []
    for rel in root.findall(relationship_tag):
        rel_id = rel.attrib.get("Id", "")
        if rel_id.startswith("rId"):
            try:
                existing_ids.append(int(rel_id[3:]))
            except ValueError:
                pass
    new_rid = f"rId{max(existing_ids, default=0) + 1}"
    ET.SubElement(
        root,
        relationship_tag,
        {
            "Id": new_rid,
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            "Target": image_target,
        },
    )
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes, new_rid


def patch_content_types(content_types_bytes: bytes) -> bytes:
    root = ET.fromstring(content_types_bytes.decode("utf-8-sig"))
    default_tag = "{http://schemas.openxmlformats.org/package/2006/content-types}Default"
    has_png = any(
        node.attrib.get("Extension", "").lower() == "png"
        for node in root.findall(default_tag)
    )
    if not has_png:
        ET.SubElement(
            root,
            default_tag,
            {"Extension": "png", "ContentType": "image/png"},
        )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_pptx() -> None:
    ensure_inputs()
    temp_path = DECK_PATH.with_suffix(".tmp.pptx")

    with ZipFile(DECK_PATH, "r") as zin, ZipFile(temp_path, "w", ZIP_DEFLATED) as zout:
        content_types = patch_content_types(zin.read("[Content_Types].xml"))
        zout.writestr("[Content_Types].xml", content_types)

        for item in zin.infolist():
            name = item.filename
            if name == "[Content_Types].xml":
                continue
            if name.startswith("ppt/slides/slide") and name.endswith(".xml"):
                continue
            if name.startswith("ppt/slides/_rels/slide") and name.endswith(".xml.rels"):
                continue
            zout.writestr(item, zin.read(name))

        for slide_index, image_path in enumerate(SLIDE_IMAGES, start=1):
            rel_path = f"ppt/slides/_rels/slide{slide_index}.xml.rels"
            rel_bytes = zin.read(rel_path)
            image_target = f"../media/image{slide_index}.png"
            patched_rels, new_rid = patch_relationships(rel_bytes, image_target)
            zout.writestr(rel_path, patched_rels)
            zout.writestr(f"ppt/slides/slide{slide_index}.xml", build_slide_xml(new_rid))
            zout.writestr(f"ppt/media/image{slide_index}.png", image_path.read_bytes())

    temp_path.replace(DECK_PATH)


def main() -> None:
    build_pptx()
    print(f"Wrote {DECK_PATH}")


if __name__ == "__main__":
    main()
