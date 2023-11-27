#include "aboba.h"

#include "ui_aboba.h"

aboba::aboba(s21::MLPController& w_controller, QWidget* parent)
    : QWidget(parent), ui(new Ui::aboba), w_controller_(w_controller) {
  ui->setupUi(this);
  Refresh();
}

aboba::~aboba() { delete ui; }

void aboba::on_DeleteAboba_clicked() {
  if (ui->AbobaView->currentItem()) {
    size_t index = ui->AbobaView->row(ui->AbobaView->currentItem());
    w_controller_.DeleteAboba(index);
    Refresh();
  }
}

void aboba::on_CreateAboba_clicked() {
  try {
    w_controller_.AddAboba(ui->MLPType->currentText() == "Matrix"
                               ? s21::MLPCore::kMatrix
                               : s21::MLPCore::kGraph,
                           ui->Topology->text().toStdString(),
                           std::string(ui->AF->currentText().toStdString()));
    auto abobas = w_controller_.GetMLPsInfo();
    ui->AbobaView->addItem(abobas.back().c_str());
  } catch (const std::exception& e) {
    ui->ErrorWindow->setText(QString(e.what()));
  }
}

void aboba::on_LoadAboba_clicked() {
  try {
    QString filename(QFileDialog::getOpenFileName(
        this, "Open File", static_cast<QDir>(QDir::homePath()).absolutePath(),
        "txt file (*.txt)"));

    w_controller_.LoadAboba(filename.toStdString());
    auto abobas = w_controller_.GetMLPsInfo();
    ui->AbobaView->addItem(abobas.back().c_str());
  } catch (const std::exception& e) {
    ui->ErrorWindow->setText(QString(e.what()));
  }
}

void aboba::Refresh() {
  ui->AbobaView->clear();
  auto abobas = w_controller_.GetMLPsInfo();
  for (const auto& s : abobas) {
    ui->AbobaView->addItem(s.c_str());
  }
}
