import { createTheme } from '@mui/material/styles';

export const themeOptions = {
    palette: {
        type: 'dark',
        primary: {
            main: '#491f7a',
        },
        secondary: {
            main: '#f50057',
        },
    },
};

export default createTheme(themeOptions);