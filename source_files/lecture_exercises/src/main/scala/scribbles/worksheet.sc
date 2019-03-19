object worksheet {
  def f_scp_treatment(dt__scp__class: java.lang.String, tmp__ma_count_of_ns: java.lang.Long, tmp__ma_count_of_unn: java.lang.Long, dt__margin_agreement__ref: String) {

    val v_ini_default = "M"
    val v_ini_used = v_ini_default

    val ma_count_of_ns = Option(Long.unbox(tmp__ma_count_of_ns))
    val ma_count_of_unn = Option(Long.unbox(tmp__ma_count_of_unn))

    val t_treatment = (v_ini_used, Option(dt__scp__class), ma_count_of_ns, ma_count_of_unn, Option(dt__margin_agreement__ref)) match {
      case (_, Some("NS"), Some(1d), Some(0d), _) => ("MY", dt__margin_agreement__ref)
      case (_, Some("SYNS"), Some(1d), Some(0d), _) => ("MY", dt__margin_agreement__ref)
      case (_, Some("NS"), _, _, _) if (ma_count_of_ns.exists(_ > 1) && ma_count_of_unn.exists(_ > 0)) => ("MN", dt__margin_agreement__ref)
      case (_, Some("SYNS"), _, _, _) if (ma_count_of_ns.exists(_ > 1) && ma_count_of_unn.exists(_ > 0)) => ("MN", dt__margin_agreement__ref)
      case (_, Some("NS"), _, _, _) => ("N", null)
      case (_, Some("SYNS"), _, _, _) => ("N", null)
      case ("M", Some("TR"), Some(1d), Some(0d), _) => ("MY", dt__margin_agreement__ref)
      case ("U", Some("TR"), Some(1d), Some(0d), _) => ("MN", dt__margin_agreement__ref)
      case (_, Some("TR"), _, _, _) if (ma_count_of_ns.exists(_ > 1) && ma_count_of_unn.exists(_ > 0)) => ("MN", dt__margin_agreement__ref)
      case (_, Some("TR"), _, _, _) => ("N", null)
      case (_, _, _, _) => (null, null)
    }

  }
}