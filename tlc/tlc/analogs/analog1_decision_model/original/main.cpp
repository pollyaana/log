#include <QApplication>
#include <StartMenu.h>

int main(int argc, char *argv[]) {
	QApplication a(argc, argv);

	StartMenu startMenu;
	startMenu.show();

	return a.exec();
}