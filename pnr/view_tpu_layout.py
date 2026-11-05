import pya

mw = pya.MainWindow.instance()
if mw:
    view = mw.current_view()
    if not view:
        mw.create_layout(0)
        view = mw.current_view()
        
    def_file = "/mnt/c/Users/surya/OneDrive/Documents/COLLEGE_RESEARCH/apex-tpu/pnr/apex_tpu_top.def"
    lef_files = [
        "/opt/orfs/flow/platforms/nangate45/lef/NangateOpenCellLibrary.tech.lef",
        "/opt/orfs/flow/platforms/nangate45/lef/NangateOpenCellLibrary.macro.mod.lef"
    ]
    lyp_file = "/mnt/c/Users/surya/OneDrive/Documents/COLLEGE_RESEARCH/apex-tpu/pnr/FreePDK45.lyp"

    opt = pya.LoadLayoutOptions()
    opt.lefdef_config.lef_files = lef_files
    opt.lefdef_config.read_lef_with_def = True
    opt.dbr = 0.0005

    print("Loading DEF with LEFs...")
    view.load_layout(def_file, opt, 0)
    print("Loading layer properties...")
    view.load_layer_props(lyp_file)
    view.zoom_fit()
    view.max_hier()
    print("Ready!")
