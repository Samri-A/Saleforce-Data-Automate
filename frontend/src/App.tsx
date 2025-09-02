import { useState } from "react";
import {
  Box,
  CssBaseline,
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  AppBar,
  Button,
  Typography,
  IconButton,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import ChatPage from "./pages/ChatPage";
import DashboardPage from "./pages/DashboardPage";
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline";
import CloseIcon from "@mui/icons-material/Close";
import AssessmentOutlinedIcon from "@mui/icons-material/AssessmentOutlined";
import CalendarTodayOutlinedIcon from '@mui/icons-material/CalendarTodayOutlined';


const drawerWidth = 240;

function App() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [selectedPage, setSelectedPage] = useState("chat");

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };
  const drawer = (
  <div>
    <Toolbar sx={{ display: "flex", justifyContent: "space-between" }}>
      <Typography sx={{ fontFamily: "cursive" }} noWrap>
        Explore
      </Typography>
      <IconButton
        edge="end"
        color="inherit"
        onClick={handleDrawerToggle}
        sx={{ display: { sm: "none" } }}
      >
        <CloseIcon />
      </IconButton>
    </Toolbar>
    <List>
      <ListItemButton onClick={() => setSelectedPage("chat")} sx={{ color: selectedPage === "chat" ? "grey" : "white" }}>
        <ListItemIcon>
          <ChatBubbleOutlineIcon
            
          />
        </ListItemIcon>
        <ListItemText primary="Chat" />
      </ListItemButton>

      <ListItemButton onClick={() => setSelectedPage("dashboard")} sx={{ color: selectedPage === "dashboard" ? "grey" : "white" }}>
        <ListItemIcon>
          <AssessmentOutlinedIcon
            
          />
        </ListItemIcon>
        <ListItemText primary="Dashboard" />
      </ListItemButton>
    </List>
  </div>
);



  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
          bgcolor: "black",
          color: "text.primary",
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: "none" } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography noWrap sx={{fontFamily : "cursive"}}>
            SaleforceAnalysis
          </Typography>
          <Box marginLeft={'auto'} >
          <Button variant="contained" sx={{ textTransform: "none"}} >
            <CalendarTodayOutlinedIcon/>
            <Typography fontSize={'14px'}>View Report</Typography>
          </Button>
          </Box>
        </Toolbar>
      </AppBar>

      <Box component="nav" sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}>
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }} 
          sx={{
            display: { xs: "block", sm: "none" },
            "& .MuiDrawer-paper": {
              width: drawerWidth,
              boxSizing: "border-box",
              bgcolor: "background.default",
              color: "white",
            },
          }}
        >
          {drawer}
        </Drawer>

        <Drawer
          variant="permanent"
          sx={{
            display: { xs: "none", sm: "block" },
            "& .MuiDrawer-paper": {
              width: drawerWidth,
              boxSizing: "border-box",
              bgcolor: "background.default",
              color: "white",
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      <Box
        component="main"
        sx={{ flexGrow: 1, p: 3, bgcolor: "background.default", minHeight: "100vh" }}
      >
        <Toolbar />
        {selectedPage === "chat" ? <ChatPage /> : <DashboardPage />}
      </Box>
    </Box>
  );
}

export default App;
