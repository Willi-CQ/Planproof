# DWG input

`planproof_cad` reads DXF directly with ezdxf. It does not parse DWG itself and does not ship a DWG converter.

A `.dwg` file is accepted only when a converter is set explicitly. The interface is `DwgConverter` in `src/planproof_cad/cad/dwg_adapter.py`: a DWG path in, a DXF path out. Adapters:

- `OdaFileConverter` starts the ODA File Converter. The path comes from the parameter, `ODA_FILE_CONVERTER`, or `PATH`.
- `LibreDwgConverter` starts LibreDWG `dwg2dxf`. The path comes from the parameter, `DWG2DXF`, or `PATH`.
- `NoConverter` is the default. Nothing is searched or started. A DWG input then fails and points at this note.

Choose a converter with the configuration key `converter` (`none`, `oda`, `libredwg`, or an object that implements the protocol), the environment variable `CQPLAN_DWG_CONVERTER`, or `python -m planproof_cad.cad run --converter`.

The ODA File Converter and LibreDWG stay outside this package. Their licenses are those of the programs that get installed separately. ODA membership terms restrict non-members to non-commercial use of the File Converter. Check the current ODA pages before using it.
