import * as React from "react";
import TextField from "@mui/material/TextField";
import Autocomplete from "@mui/material/Autocomplete";
import Box from "@mui/material/Box";
import BusinessCenterOutlinedIcon from "@mui/icons-material/BusinessCenterOutlined";
import EmojiTransportationOutlinedIcon from "@mui/icons-material/EmojiTransportationOutlined";

import categories from "./categories";
import { styled } from "@mui/material/styles";
import { useEffect, useState } from "react";

const CssTextField = styled(TextField)({
  "& label": {
    color: "white",
  },
  "& input": {
    color: "white",
  },
  "& .MuiOutlinedInput-root": {
    "& fieldset": {
      color: "red",
    },
  },
});

export default function Form({ onValueChange }) {
  const [city, setCity] = useState("");

  const [value, setValue] = React.useState(categories[0]);
  const [inputValue, setInputValue] = React.useState("");

  const handleChange = (event) => {
    setCity(event.target.value);
  };

  useEffect(() => {
    onValueChange({
      city: city,
      category: value?.label,
    });
  }, [value, city]);

  return (
    <React.Fragment>
      <Box sx={{ display: "flex", alignItems: "flex-end" }}>
        <EmojiTransportationOutlinedIcon
          sx={{ color: "white", mr: 1, my: 0.5 }}
        />
        <CssTextField
          color={"primary"}
          size={"small"}
          variant="standard"
          placeholder="City"
          value={city}
          onChange={handleChange}
        />
      </Box>
      <Box sx={{ display: "flex", alignItems: "flex-end" }}>
        <BusinessCenterOutlinedIcon sx={{ color: "white", mr: 1, my: 0.5 }} />
        <Autocomplete
          value={value}
          onChange={(event, newValue) => {
            setValue(newValue);
          }}
          inputValue={inputValue}
          onInputChange={(event, newInputValue) => {
            setInputValue(newInputValue);
          }}
          disablePortal
          id="category selector"
          options={categories}
          sx={{ width: 300, marginTop: "-2px" }}
          renderInput={(params) => (
            <CssTextField
              {...params}
              color={"primary"}
              size={"small"}
              placeholder="Category"
              variant="standard"
            />
          )}
        />
      </Box>
    </React.Fragment>
  );
}
