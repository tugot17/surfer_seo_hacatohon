import * as React from "react";
import Fab from "@mui/material/Fab";
import BusinessIcon from "@mui/icons-material/Business";
import { Tooltip } from "@mui/material";

const BusinessPoint = ({ name, lat, lng, $hover, zIndex }) => {
  return (
    <Tooltip title={name}>
      <Fab aria-label="save" color="primary">
        <BusinessIcon fontSize={"large"} />
      </Fab>
    </Tooltip>
  );
};

export default BusinessPoint;
