import * as React from "react";
import { styled, createTheme, ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import MuiDrawer from "@mui/material/Drawer";
import Box from "@mui/material/Box";
import MuiAppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import List from "@mui/material/List";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import Badge from "@mui/material/Badge";
import Container from "@mui/material/Container";
import CircularProgress from "@mui/material/CircularProgress";
import MenuIcon from "@mui/icons-material/Menu";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ScreenSearchDesktopIcon from "@mui/icons-material/ScreenSearchDesktop";
import NotificationsIcon from "@mui/icons-material/Notifications";
import { green } from "@mui/material/colors";
import logo from "./logo.png";
import Form from "../Form/Form";
import { useState } from "react";
import axios from "axios";

function Copyright(props) {
  return (
    <Typography
      variant="body2"
      color="text.secondary"
      align="center"
      {...props}
    >
      {"Copyright © Surfer PWR Hackaton"}
      {new Date().getFullYear()}
      {"."}
    </Typography>
  );
}

const AppBar = styled(MuiAppBar, {
  shouldForwardProp: (prop) => prop !== "open",
})(({ theme, open }) => ({
  zIndex: theme.zIndex.drawer + 1,
}));

function LayoutContent({ children, handleDataChange }) {
  const [form, setForm] = useState({ city: "", category: "" });
  const [loading, setLoading] = useState(false);

  const handleForm = (newForm) => {
    setForm(newForm);
  };
  const handleFormSearch = async () => {
    setLoading(true);
    const params = [
      ["category", form.category],
      ["city", form.city],
      ["grid_size", 10],
    ];

    const url = new URL("http://192.168.43.42:8000");
    url.search = new URLSearchParams(params).toString();
    const headers = {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
    };

    axios.defaults.headers.post["Access-Control-Allow-Origin"] = "*";
    axios
      .get("http://192.168.43.42:8000/" + url.search, { headers })
      .then((response) => {
        console.log("Success ========>", response.data);
        handleDataChange(response.data);
        setLoading(false);
      })
      .catch((error) => {
        console.log("Error ========>", error);
        setLoading(false);
      });
  };
  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />
      <AppBar position="absolute">
        <Toolbar
          sx={{
            pr: "24px", // keep right padding when drawer closed
          }}
          variant={"dense"}
          color={"secondary"}
        >
          <img
            src={logo}
            alt={logo}
            style={{ height: "24px", marginRight: "12px" }}
          />
          <Typography
            component="p"
            variant="body"
            color="inherit"
            noWrap
            sx={{ flexGrow: 1 }}
          >
            Surfer Business Placer
          </Typography>
          <Form onValueChange={handleForm} />

          <Box sx={{ m: 1, position: "relative", marginLeft: 2 }}>
            <IconButton
              color="inherit"
              onClick={handleFormSearch}
              disabled={form.city === "" || form.category == null || loading}
            >
              <ScreenSearchDesktopIcon fontSize={"large"} />
            </IconButton>

            {loading && (
              <CircularProgress
                size={54}
                sx={{
                  color: green[500],
                  position: "absolute",
                  top: -2,
                  left: -2,
                  zIndex: 1,
                }}
              />
            )}
          </Box>
        </Toolbar>
      </AppBar>
      <Box
        component="main"
        sx={{
          backgroundColor: (theme) =>
            theme.palette.mode === "light"
              ? theme.palette.grey[100]
              : theme.palette.grey[900],
          flexGrow: 1,
          height: "100vh",
          overflow: "auto",
          padding: 0,
        }}
      >
        <Toolbar variant={"dense"} />
        <Container
          maxWidth={false}
          style={{
            paddingLeft: 0,
            paddingRight: 0,
            height: "calc(100vh - 48px)",
          }}
        >
          {children}
        </Container>
      </Box>
    </Box>
  );
}

export default function Layout({ children, handleDataChange }) {
  return <LayoutContent handleDataChange={handleDataChange}>{children}</LayoutContent>;
}
